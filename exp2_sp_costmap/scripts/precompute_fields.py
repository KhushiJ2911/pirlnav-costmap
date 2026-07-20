"""Offline precompute of costmap geodesic fields.

Builds one field per (scene, category) from the shipped navmesh, each in a
hard-terminate()-able subprocess so a degenerate find_path that would hang
training online is killed after TIMEOUT and replaced by a cheap Euclidean field.
Resumable (skips existing .npz). Fields load at train/eval time -> no online
field building -> no possible hang.

Usage: python precompute_fields.py <content_glob> <out_dir>
"""
import os, sys, gzip, json, glob, time
import queue as _queue
import numpy as np
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import costmap_field_builder as cfb

N_SLOTS = 10
TIMEOUT = 360.0


def enumerate_tasks(content_glob, out_dir):
    tasks = []
    for f in sorted(glob.glob(content_glob)):
        d = json.load(gzip.open(f))
        gbc = d.get("goals_by_category", {})
        if not d.get("episodes"):
            continue
        scene_id = d["episodes"][0]["scene_id"]
        scene_glb = os.path.basename(scene_id)
        navmesh = "data/scene_datasets/" + scene_id.replace(".basis.glb", ".basis.navmesh")
        for key, goalobjs in gbc.items():
            if key.startswith(scene_glb + "_"):
                cat = key[len(scene_glb) + 1:]
            else:
                cat = goalobjs[0]["object_category"]
            goals = []
            for g in goalobjs:
                vps = g.get("view_points")
                if vps:
                    goals += [vp["agent_state"]["position"] for vp in vps]
                else:
                    goals.append(g["position"])
            out = os.path.join(out_dir, "%s__%s.npz" % (scene_glb, cat))
            tasks.append((scene_glb, cat, navmesh, np.asarray(goals, np.float32), out))
    return tasks


def save(out, pts, ds, method):
    # np.savez appends ".npz" when given a path string; pass a file handle so
    # the temp file keeps its exact name and os.replace is atomic.
    tmp = out + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, pts=pts.astype(np.float32), ds=ds.astype(np.float32),
                 method=np.array(method))
    os.replace(tmp, out)


def euclidean_fallback(out, navmesh, goals, tag):
    pts, ds = cfb.build_euclidean(navmesh, goals)
    save(out, pts, ds, tag)


def main():
    content_glob = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    tasks = enumerate_tasks(content_glob, out_dir)
    todo = [t for t in tasks if not os.path.exists(t[4])]
    print("total %d | already done %d | to build %d"
          % (len(tasks), len(tasks) - len(todo), len(todo)), flush=True)

    running = {}
    done = 0
    method_counts = {}

    def _finish(tag):
        method_counts[tag] = method_counts.get(tag, 0) + 1

    while todo or running:
        while todo and len(running) < N_SLOTS:
            task = todo.pop()
            _, _, navmesh, goals, out = task
            q = mp.Queue()
            p = mp.Process(target=cfb._worker, args=(navmesh, goals, q))
            p.daemon = True
            p.start()
            running[p] = (task, time.time(), q)

        for p in list(running):
            task, t0, q = running[p]
            scene_glb, cat, navmesh, goals, out = task
            got = None
            try:
                got = q.get(timeout=0.05)
            except _queue.Empty:
                got = None

            if got is not None:
                if got[0] == "ok":
                    save(out, got[1], got[2], "geodesic"); _finish("geodesic")
                else:
                    euclidean_fallback(out, navmesh, goals, "euclidean_err"); _finish("euclidean_err")
                    print("  ERR->euclidean:", scene_glb, cat, got[1], flush=True)
                p.join(timeout=3)
                if p.is_alive():
                    p.terminate(); p.join()
                del running[p]; done += 1
            elif time.time() - t0 > TIMEOUT:
                p.terminate(); p.join()
                print("  TIMEOUT->euclidean: %s %s (%.0fs)" % (scene_glb, cat, time.time() - t0), flush=True)
                euclidean_fallback(out, navmesh, goals, "euclidean_timeout"); _finish("euclidean_timeout")
                del running[p]; done += 1
            elif not p.is_alive():
                # exited without a result; give the queue one more chance
                try:
                    got = q.get(timeout=1.0)
                except _queue.Empty:
                    got = None
                if got is not None and got[0] == "ok":
                    save(out, got[1], got[2], "geodesic"); _finish("geodesic")
                else:
                    euclidean_fallback(out, navmesh, goals, "euclidean_dead"); _finish("euclidean_dead")
                    print("  DEAD->euclidean:", scene_glb, cat, flush=True)
                p.join(); del running[p]; done += 1

            if done and done % 50 == 0:
                print("progress %d  %s" % (done, method_counts), flush=True)

        time.sleep(0.2)

    print("DONE. counts:", method_counts, flush=True)


if __name__ == "__main__":
    main()
