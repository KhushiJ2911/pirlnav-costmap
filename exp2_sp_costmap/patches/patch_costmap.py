import io, re, sys

path = "pirlnav/task/costmap_sensor.py"
src = open(path).read()
orig = src

# 1) module imports: add signal + threading after "import time"
if "import signal" not in src:
    src = src.replace(
        "import time\nfrom collections import OrderedDict",
        "import time\nimport signal\nimport threading\nfrom collections import OrderedDict",
        1,
    )

# 2) module-level timeout helper, inserted just before the class decorator
helper = (
    'class _FieldTimeout(Exception):\n'
    '    pass\n'
    '\n'
    '\n'
    'def _raise_field_timeout(signum, frame):\n'
    '    raise _FieldTimeout()\n'
    '\n'
    '\n'
)
if "_FieldTimeout" not in src:
    src = src.replace(
        '@registry.register_sensor(name="CostmapSensor")',
        helper + '@registry.register_sensor(name="CostmapSensor")',
        1,
    )

# 3) replace the whole _build_field method (from its def to the next def)
lines = src.splitlines(keepends=True)
start = None
for i, ln in enumerate(lines):
    if ln.strip().startswith("def _build_field(self, goals):"):
        start = i
        break
assert start is not None, "could not find _build_field"
# find next top-level method def at the same indentation (4 spaces)
indent = len(lines[start]) - len(lines[start].lstrip())
end = None
for j in range(start + 1, len(lines)):
    s = lines[j]
    if s.strip().startswith("def ") and (len(s) - len(s.lstrip())) == indent:
        end = j
        break
assert end is not None, "could not find method after _build_field"

new_method = '''    def _build_field(self, goals):
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
'''

lines[start:end] = [new_method]
src = "".join(lines)

open(path, "w").write(src)

# syntax check
import ast
ast.parse(open(path).read())
print("patched OK; changed:", orig != src)
