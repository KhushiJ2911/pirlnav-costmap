import ast

path = "pirlnav/task/costmap_sensor.py"
src = open(path).read()
orig = src

# 1) import os
if "\nimport os\n" not in src:
    src = src.replace("import time\n", "import time\nimport os\n", 1)

# 2) field cache dir in __init__
anchor_cap = "        self._field_cache_cap = 512\n"
assert anchor_cap in src, "cap anchor missing"
if "_field_cache_dir" not in src:
    src = src.replace(
        anchor_cap,
        anchor_cap
        + '        # Precomputed geodesic fields (offline, hard-timeout protected)\n'
        + '        # live here, keyed "<scene_glb>__<category>.npz". Loading them at\n'
        + '        # runtime avoids online field building, which can hang training on\n'
        + '        # degenerate navmeshes. Override dir via COSTMAP_FIELD_CACHE.\n'
        + '        self._field_cache_dir = os.environ.get(\n'
        + '            "COSTMAP_FIELD_CACHE", "data/costmap_field_cache"\n'
        + '        )\n',
        1,
    )

# 3) loader + euclidean fallback methods, inserted before get_observation
loader = '''    def _load_precomputed(self, scene_id, category):
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

    def get_observation(self, observations, episode, *args, **kwargs):'''

anchor_getobs = "    def get_observation(self, observations, episode, *args, **kwargs):"
assert anchor_getobs in src, "get_observation anchor missing"
if "_load_precomputed" not in src:
    src = src.replace(anchor_getobs, loader, 1)

# 4) replace online build in the else-branch with load-from-cache
old_build = '''                t0 = time.time()
                self._build_field(goals)
                logger.info(
                    "CostmapSensor: built field for {} in {:.2f}s "
                    "({} grid pts)".format(
                        key, time.time() - t0, len(self._field_d)
                    )
                )
'''
new_build = '''                cat = getattr(episode, "object_category", None)
                loaded = self._load_precomputed(episode.scene_id, cat)
                if loaded is not None:
                    self._tree, self._field_d = loaded
                else:
                    logger.warning(
                        "CostmapSensor: no precomputed field for {}; "
                        "using Euclidean fallback.".format(key)
                    )
                    self._euclidean_field(goals)
'''
assert old_build in src, "online-build anchor missing"
src = src.replace(old_build, new_build, 1)

open(path, "w").write(src)
ast.parse(open(path).read())
print("sensor patched OK; changed:", orig != src)
