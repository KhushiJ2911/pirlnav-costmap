"""Standalone geodesic-field builder for the costmap sensor.

Loads a scene's .navmesh directly (no full Simulator, no CUDA) and builds the
exact same field CostmapSensor._build_field produces online, so fields can be
precomputed offline in hard-terminate()-able subprocesses instead of built
online (where a degenerate find_path can spin forever and hang training).

GRID_RES=0.2 and the >15000-point adaptive coarsening match the online sensor.
"""
import numpy as np
import habitat_sim

GRID_RES = 0.2
ADAPTIVE_CAP = 15000.0


def _adaptive_step(lo, hi, res=GRID_RES):
    step = res
    span_x = float(hi[0] - lo[0])
    span_z = float(hi[2] - lo[2])
    if (span_x / step) * (span_z / step) > ADAPTIVE_CAP:
        step = ((span_x * span_z) / ADAPTIVE_CAP) ** 0.5
    return step


def load_pathfinder(navmesh_path):
    pf = habitat_sim.PathFinder()
    pf.load_nav_mesh(navmesh_path)
    if not pf.is_loaded:
        raise RuntimeError("navmesh failed to load: %s" % navmesh_path)
    return pf


def build_geodesic(navmesh_path, goals):
    """Exact replica of CostmapSensor._build_field (geodesic)."""
    pf = load_pathfinder(navmesh_path)
    goals = np.asarray(goals, np.float32)

    lo, hi = pf.get_bounds()
    y0 = float(goals[0][1])
    step = _adaptive_step(lo, hi)

    sp = habitat_sim.MultiGoalShortestPath()
    sp.requested_ends = np.asarray(goals, dtype=np.float32)

    pts = []
    ds = []
    for x in np.arange(lo[0], hi[0], step):
        for z in np.arange(lo[2], hi[2], step):
            p = pf.snap_point(np.array([x, y0, z], np.float32))
            if np.any(np.isnan(p)):
                continue
            sp.requested_start = np.asarray(p, np.float32)
            d = sp.geodesic_distance if pf.find_path(sp) else np.inf
            if np.isfinite(d):
                pts.append([p[0], p[2]])
                ds.append(d)
    return np.asarray(pts, np.float32), np.asarray(ds, np.float32)


def build_euclidean(navmesh_path, goals):
    """Fallback: min planar (xz) distance to any goal. No find_path -> no hang."""
    pf = load_pathfinder(navmesh_path)
    goals = np.asarray(goals, np.float32)
    gx = goals[:, 0]
    gz = goals[:, 2]

    lo, hi = pf.get_bounds()
    y0 = float(goals[0][1])
    step = _adaptive_step(lo, hi)

    pts = []
    ds = []
    for x in np.arange(lo[0], hi[0], step):
        for z in np.arange(lo[2], hi[2], step):
            p = pf.snap_point(np.array([x, y0, z], np.float32))
            if np.any(np.isnan(p)):
                continue
            dmin = float(np.min(np.hypot(gx - p[0], gz - p[2])))
            pts.append([p[0], p[2]])
            ds.append(dmin)
    return np.asarray(pts, np.float32), np.asarray(ds, np.float32)


def _worker(navmesh_path, goals, q):
    try:
        pts, ds = build_geodesic(navmesh_path, goals)
        q.put(("ok", pts, ds))
    except Exception as e:  # pragma: no cover
        q.put(("err", str(e), None))
