"""Online WayPixel costmap sensor (Aditya's method, fast field-lookup version)."""

import time
import os
import signal
import threading
from collections import OrderedDict

import numpy as np
import habitat_sim
import quaternion  # noqa
import cv2

from gym import spaces
from habitat import logger
from scipy.spatial import cKDTree

from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes

# Opt-in hang diagnostics: with COSTMAP_FAULTHANDLER set, dump every
# thread's Python stack every N seconds to stderr. During a hang the
# same stack repeats -> pinpoints exactly where it is stuck (works
# without ptrace, which this container lacks).
if os.environ.get("COSTMAP_FAULTHANDLER"):
    import sys as _sys
    import faulthandler as _fh
    _fh.dump_traceback_later(
        int(os.environ.get("COSTMAP_FAULTHANDLER", "300")) or 300,
        repeat=True, file=_sys.stderr,
    )


class _FieldTimeout(Exception):
    pass


def _raise_field_timeout(signum, frame):
    raise _FieldTimeout()


@registry.register_sensor(name="CostmapSensor")
class CostmapSensor(Sensor):
    cls_uuid = "costmap"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim

        self.out_h = config.OUT_HEIGHT
        self.out_w = config.OUT_WIDTH

        self.hfov = float(config.HFOV)
        self.sensor_height = float(config.SENSOR_HEIGHT)

        self.min_depth = float(config.MIN_DEPTH)
        self.max_depth = float(config.MAX_DEPTH)
        self.normalize_depth = bool(config.NORMALIZE_DEPTH)

        self.grid_res = float(config.GRID_RES)
        self.max_cost = float(config.MAX_COST)

        self._tree = None
        self._field_d = None
        self._episode_id = None
        self._K = None

        # The geodesic field depends only on (scene, goal set) = (scene, object
        # category): <= n_categories distinct fields per scene. Cache them so the
        # expensive _build_field runs once per (scene, category), not per episode.
        self._field_cache = OrderedDict()
        # Cover all (scene, category) fields a worker can encounter in full
        # multi-scene training (80 scenes x 6 categories = 480). With the old
        # cap of 16 the LRU thrashed under multi-scene training — evicting and
        # rebuilding expensive geodesic fields every few steps, stalling
        # throughput. Fields are tiny (~200KB), so caching all is cheap.
        self._field_cache_cap = 512
        # Precomputed geodesic fields (offline, hard-timeout protected)
        # live here, keyed "<scene_glb>__<category>.npz". Loading them at
        # runtime avoids online field building, which can hang training on
        # degenerate navmeshes. Override dir via COSTMAP_FIELD_CACHE.
        self._field_cache_dir = os.environ.get(
            "COSTMAP_FIELD_CACHE", "data/costmap_field_cache"
        )

        super().__init__(config=config)

    def _get_uuid(self, *a, **k):
        return self.cls_uuid

    def _get_sensor_type(self, *a, **k):
        return SensorTypes.DEPTH

    def _get_observation_space(self, *a, **k):
        return spaces.Box(
            low=0.0,
            high=np.finfo(np.float32).max,
            shape=(self.out_h, self.out_w, 1),
            dtype=np.float32,
        )

    def _geo(self, a, b):
        sp = habitat_sim.ShortestPath()
        sp.requested_start = np.asarray(a, np.float32)
        sp.requested_end = np.asarray(b, np.float32)

        return (
            sp.geodesic_distance
            if self._sim.pathfinder.find_path(sp)
            else np.inf
        )

    def _build_field(self, goals):
        pf = self._sim.pathfinder

        lo, hi = pf.get_bounds()
        y0 = goals[0][1]

        goals_xz = np.asarray([[g[0], g[2]] for g in goals], np.float32)

        pts = []
        ds = []

        # One multi-goal query per grid point (habitat computes min over all
        # goals internally) instead of len(goals) separate ShortestPath queries.
        sp = habitat_sim.MultiGoalShortestPath()
        sp.requested_ends = np.asarray(goals, dtype=np.float32)

        # Adaptive grid step: keep the fine grid_res for normal scenes, but
        # coarsen for monster scenes so a single field build cannot blow up.
        step = self.grid_res
        span_x = float(hi[0] - lo[0]); span_z = float(hi[2] - lo[2])
        if (span_x / step) * (span_z / step) > 15000.0:
            step = ((span_x * span_z) / 15000.0) ** 0.5

        # Field-build watchdog. Some HM3D navmeshes drive habitat's pathfinder
        # into a minutes-to-hours spin on a single degenerate find_path (a ~9h
        # hang was observed on scene 00475-g7hUFVNac26). Two guards: (1) a
        # wall-clock budget checked every grid row, and (2) a SIGALRM hard cap
        # (main thread only). Either drops us to a cheap Euclidean fallback
        # field so a pathological scene can never stall training again.
        budget_s = float(getattr(self, "_field_build_budget_s", 240.0))
        timed_out = False
        t0 = time.time()

        use_alarm = threading.current_thread() is threading.main_thread()
        old_handler = None
        if use_alarm:
            old_handler = signal.signal(signal.SIGALRM, _raise_field_timeout)
            signal.setitimer(signal.ITIMER_REAL, budget_s + 30.0)

        try:
            for x in np.arange(lo[0], hi[0], step):
                if time.time() - t0 > budget_s:
                    timed_out = True
                    break
                for z in np.arange(lo[2], hi[2], step):
                    p = pf.snap_point(np.array([x, y0, z], np.float32))
                    if np.any(np.isnan(p)):
                        continue
                    sp.requested_start = np.asarray(p, np.float32)
                    d = sp.geodesic_distance if pf.find_path(sp) else np.inf
                    if np.isfinite(d):
                        pts.append([p[0], p[2]])
                        ds.append(d)
        except _FieldTimeout:
            timed_out = True
        finally:
            if use_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)

        if timed_out or len(pts) < 4:
            logger.warning(
                "CostmapSensor: geodesic field build exceeded %.0fs budget; "
                "falling back to Euclidean field." % budget_s
            )
            pts = []
            ds = []
            for x in np.arange(lo[0], hi[0], step):
                for z in np.arange(lo[2], hi[2], step):
                    p = pf.snap_point(np.array([x, y0, z], np.float32))
                    if np.any(np.isnan(p)):
                        continue
                    dmin = float(
                        np.min(
                            np.hypot(
                                goals_xz[:, 0] - p[0],
                                goals_xz[:, 1] - p[2],
                            )
                        )
                    )
                    pts.append([p[0], p[2]])
                    ds.append(dmin)

        self._tree = cKDTree(np.asarray(pts, np.float32))
        self._field_d = np.asarray(ds, np.float32)
    def _K_for(self, H, W):
        fx = fy = (W / 2.0) / np.tan(
            np.radians(self.hfov) / 2.0
        )

        return np.array(
            [
                [fx, 0, W / 2.0],
                [0, fy, H / 2.0],
                [0, 0, 1],
            ],
            np.float32,
        )

    def _backproject(self, depth, st):
        H, W = depth.shape

        if self._K is None:
            self._K = self._K_for(H, W)
            # Ray directions depend only on the intrinsics/resolution, not the
            # depth: precompute once instead of rebuilding mgrid every step.
            v, u = np.mgrid[0:H, 0:W]
            self._xr = ((u - self._K[0, 2]) / self._K[0, 0]).astype(np.float32)
            self._yr = ((v - self._K[1, 2]) / self._K[1, 1]).astype(np.float32)

        Z = depth
        X = self._xr * Z
        Y = self._yr * Z

        pts = np.stack(
            [X, -Y, -Z],
            axis=-1,
        ).reshape(-1, 3)  # RDF -> Habitat RUB

        R = quaternion.as_rotation_matrix(st.rotation)
        cam = np.asarray(st.position, np.float32)

        return ((R @ pts.T).T + cam).reshape(H, W, 3)

    def _load_precomputed(self, scene_id, category):
        if not category:
            return None
        fname = "{}__{}.npz".format(os.path.basename(scene_id), category)
        path = os.path.join(self._field_cache_dir, fname)
        if not os.path.exists(path):
            return None
        try:
            data = np.load(path)
            pts = np.asarray(data["pts"], np.float32)
            ds = np.asarray(data["ds"], np.float32)
        except Exception:
            return None
        if len(pts) < 1:
            return None
        return cKDTree(pts), ds

    def _euclidean_field(self, goals):
        # Safe fallback if a precomputed field is missing: min planar distance
        # to any goal. Uses the navmesh only to snap grid points (no find_path,
        # so it can never hang).
        pf = self._sim.pathfinder
        lo, hi = pf.get_bounds()
        goals = np.asarray(goals, np.float32)
        gx = goals[:, 0]; gz = goals[:, 2]
        y0 = float(goals[0][1])
        step = self.grid_res
        span_x = float(hi[0] - lo[0]); span_z = float(hi[2] - lo[2])
        if (span_x / step) * (span_z / step) > 15000.0:
            step = ((span_x * span_z) / 15000.0) ** 0.5
        pts = []; ds = []
        for x in np.arange(lo[0], hi[0], step):
            for z in np.arange(lo[2], hi[2], step):
                p = pf.snap_point(np.array([x, y0, z], np.float32))
                if np.any(np.isnan(p)):
                    continue
                pts.append([p[0], p[2]])
                ds.append(float(np.min(np.hypot(gx - p[0], gz - p[2]))))
        self._tree = cKDTree(np.asarray(pts, np.float32))
        self._field_d = np.asarray(ds, np.float32)

    def get_observation(self, observations, episode, *args, **kwargs):
        if episode.episode_id != self._episode_id:
            self._episode_id = episode.episode_id
            self._K = None  # depth intrinsics may differ across resets

            key = (
                episode.scene_id,
                getattr(episode, "object_category", None)
                or episode.episode_id,
            )
            cached = self._field_cache.get(key)
            if cached is not None:
                self._tree, self._field_d = cached
                self._field_cache.move_to_end(key)
            else:
                goals = []
                for g in episode.goals:
                    vps = getattr(g, "view_points", None)
                    if vps:
                        goals += [
                            np.asarray(vp.agent_state.position, np.float32)
                            for vp in vps
                        ]
                    else:
                        goals.append(np.asarray(g.position, np.float32))

                cat = getattr(episode, "object_category", None)
                loaded = self._load_precomputed(episode.scene_id, cat)
                if loaded is not None:
                    self._tree, self._field_d = loaded
                else:
                    logger.warning(
                        "CostmapSensor: no precomputed field for {}; "
                        "using Euclidean fallback.".format(key)
                    )
                    self._euclidean_field(goals)
                self._field_cache[key] = (self._tree, self._field_d)
                if len(self._field_cache) > self._field_cache_cap:
                    self._field_cache.popitem(last=False)

        depth = np.asarray(observations["depth"], np.float32)

        if depth.ndim == 3:
            depth = depth[:, :, 0]

        if self.normalize_depth:
            depth = (
                depth * (self.max_depth - self.min_depth)
                + self.min_depth
            )

        # Compute at the OUTPUT resolution: backprojection + KD-tree lookup cost
        # scales with pixel count, and the geodesic field is low-frequency, so
        # downsampling depth first (instead of resizing the result) is ~free
        # accuracy-wise and cuts the per-step cost by (in/out)^2.
        if depth.shape != (self.out_h, self.out_w):
            depth = cv2.resize(
                depth,
                (self.out_w, self.out_h),
                interpolation=cv2.INTER_NEAREST,
            )

        world = self._backproject(
            depth,
            self._sim.get_agent_state().sensor_states["depth"],
        )

        H, W = depth.shape

        out = np.full(
            H * W,
            self.max_cost,
            np.float32,
        )

        valid = (
            (
                (depth > self.min_depth + 1e-3)
                & (depth < self.max_depth - 1e-3)
                & np.isfinite(depth)
            )
            .reshape(-1)
        )

        if valid.any():
            _, idx = self._tree.query(
                world[:, :, [0, 2]]
                .reshape(-1, 2)[valid]
            )

            out[valid] = np.clip(
                self._field_d[idx],
                0,
                self.max_cost,
            )

        out = out.reshape(H, W)

        if (H, W) != (self.out_h, self.out_w):
            out = cv2.resize(
                out,
                (self.out_w, self.out_h),
                interpolation=cv2.INTER_NEAREST,
            )

        return out[:, :, None].astype(np.float32)