"""Validate the standalone builder against online-logged grid-point counts.

Online log (qk9eeNeR4vw) reported: plant=5131, bed=5135, tv_monitor=5135 pts.
If the standalone build reproduces these, the shipped .navmesh matches the
sim's runtime navmesh and the field logic is identical -> precompute is valid.
"""
import gzip, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from costmap_field_builder import build_geodesic

SCENE_CONTENT = "data/datasets/objectnav/objectnav_hm3d_sp_gen/train/content/qk9eeNeR4vw.json.gz"
EXPECTED = {"plant": 5131, "bed": 5135, "tv_monitor": 5135}


def goals_for(gbc, scene_glb, category):
    key = scene_glb + "_" + category
    goals = []
    for gobj in gbc[key]:
        vps = gobj.get("view_points")
        if vps:
            goals += [vp["agent_state"]["position"] for vp in vps]
        else:
            goals.append(gobj["position"])
    return np.asarray(goals, np.float32)


def main():
    d = json.load(gzip.open(SCENE_CONTENT))
    gbc = d["goals_by_category"]
    scene_id = d["episodes"][0]["scene_id"]  # hm3d/train/00016-qk9eeNeR4vw/...glb
    scene_glb = os.path.basename(scene_id)   # qk9eeNeR4vw.basis.glb
    navmesh = "data/scene_datasets/" + scene_id.replace(".basis.glb", ".basis.navmesh")
    print("scene_id:", scene_id)
    print("navmesh :", navmesh, "exists:", os.path.exists(navmesh))
    ok = True
    for cat, exp in EXPECTED.items():
        goals = goals_for(gbc, scene_glb, cat)
        t0 = time.time()
        pts, ds = build_geodesic(navmesh, goals)
        n = len(pts)
        match = (n == exp)
        ok = ok and match
        print("  %-11s goals=%5d  pts=%5d  expected=%5d  %s  dmin=%.2f dmax=%.2f  (%.1fs)" % (
            cat, len(goals), n, exp, "MATCH" if match else "MISMATCH",
            float(ds.min()) if n else -1, float(ds.max()) if n else -1, time.time() - t0))
    print("VALIDATION:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
