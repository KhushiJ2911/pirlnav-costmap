import ast
path = "pirlnav/task/costmap_sensor.py"
src = open(path).read()
orig = src

anchor = "from habitat.core.simulator import Sensor, SensorTypes\n"
assert anchor in src
block = (
    anchor
    + "\n"
    + "# Opt-in hang diagnostics: with COSTMAP_FAULTHANDLER set, dump every\n"
    + "# thread's Python stack every N seconds to stderr. During a hang the\n"
    + "# same stack repeats -> pinpoints exactly where it is stuck (works\n"
    + "# without ptrace, which this container lacks).\n"
    + "if os.environ.get(\"COSTMAP_FAULTHANDLER\"):\n"
    + "    import sys as _sys\n"
    + "    import faulthandler as _fh\n"
    + "    _fh.dump_traceback_later(\n"
    + "        int(os.environ.get(\"COSTMAP_FAULTHANDLER\", \"300\")) or 300,\n"
    + "        repeat=True, file=_sys.stderr,\n"
    + "    )\n"
)
if "COSTMAP_FAULTHANDLER" not in src:
    src = src.replace(anchor, block, 1)

open(path, "w").write(src)
ast.parse(open(path).read())
print("faulthandler patch OK; changed:", orig != src)
