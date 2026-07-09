"""Validate fast online costmap (geodesic field + KD-tree) vs Aditya's per-pixel costmaps."""

import argparse, os, time
import numpy as np
import habitat_sim
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRID_RES = 0.2


def geodesic(pf, a, b):
    sp = habitat_sim.ShortestPath()
    sp.requested_start = np.asarray(a, dtype=np.float32)
    sp.requested_end = np.asarray(b, dtype=np.float32)
    if pf.find_path(sp):
        return sp.geodesic_distance
    return np.inf


def build_field(pf, goal, res=GRID_RES):
    lo, hi = pf.get_bounds()
    xs = np.arange(lo[0], hi[0], res)
    zs = np.arange(lo[2], hi[2], res)

    pts, ds = [], []

    for x in xs:
        for z in zs:
            p = pf.snap_point(np.array([x, goal[1], z], dtype=np.float32))
            if np.any(np.isnan(p)):
                continue

            d = geodesic(pf, p, goal)
            if np.isfinite(d):
                pts.append([p[0], p[2]])
                ds.append(d)

    return (
        cKDTree(np.asarray(pts, dtype=np.float32)),
        np.asarray(ds, dtype=np.float32),
    )


def online_costmap(pointmap, tree, field_d):
    H, W, _ = pointmap.shape
    out = np.full((H, W), np.nan, dtype=np.float32)

    valid = np.isfinite(pointmap).all(axis=2)
    xz = pointmap[:, :, [0, 2]][valid]

    if len(xz):
        _, idx = tree.query(xz)
        out[valid] = field_d[idx]

    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--session", required=True)
    ap.add_argument("--ref_arrays", required=True)
    ap.add_argument("--navmesh", required=True)

    ap.add_argument("--goal_frame", type=int, default=70)
    ap.add_argument("--goal_u", type=int, default=160)
    ap.add_argument("--goal_v", type=int, default=120)

    ap.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=[0, 35, 70],
    )

    args = ap.parse_args()

    session = os.path.expanduser(args.session)
    ref_dir = os.path.expanduser(args.ref_arrays)

    pm_dir = os.path.join(session, "gt_pointmaps_fov90")
    out_dir = os.path.join(session, "validation")
    os.makedirs(out_dir, exist_ok=True)

    pf = habitat_sim.PathFinder()
    pf.load_nav_mesh(args.navmesh)
    assert pf.is_loaded, "navmesh failed to load"

    gpm = np.load(os.path.join(pm_dir, f"{args.goal_frame:05d}.npy"))
    goal_raw = gpm[args.goal_v, args.goal_u]
    goal = pf.snap_point(np.asarray(goal_raw, dtype=np.float32))

    print(f"goal raw={goal_raw} snapped={goal}")

    t0 = time.time()

    tree, field_d = build_field(pf, goal)

    print(
        f"geodesic field: {len(field_d)} pts in {time.time()-t0:.2f}s "
        f"(dist range {field_d.min():.2f}..{field_d.max():.2f} m)"
    )

    for f in args.frames:
        pm = np.load(os.path.join(pm_dir, f"{f:05d}.npy"))
        ref = np.load(os.path.join(ref_dir, f"{f:05d}.npy"))

        t1 = time.time()
        mine = online_costmap(pm, tree, field_d)
        dt = (time.time() - t1) * 1000

        m = np.isfinite(mine) & np.isfinite(ref)

        if m.sum() == 0:
            print(f"frame {f}: no overlapping valid pixels")
            continue

        diff = np.abs(mine[m] - ref[m])
        corr = np.corrcoef(mine[m], ref[m])[0, 1]

        print(
            f"frame {f}: {dt:.1f} ms | "
            f"MAE={diff.mean():.3f}m "
            f"median={np.median(diff):.3f}m "
            f"p90={np.percentile(diff,90):.3f}m "
            f"corr={corr:.4f}"
        )

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        vmax = np.nanpercentile(ref[m], 99)

        a0 = axs[0].imshow(ref, cmap="turbo", vmin=0, vmax=vmax)
        axs[0].set_title(f"Aditya ref (f{f})")
        plt.colorbar(a0, ax=axs[0])

        a1 = axs[1].imshow(mine, cmap="turbo", vmin=0, vmax=vmax)
        axs[1].set_title("mine (field lookup)")
        plt.colorbar(a1, ax=axs[1])

        d = np.full_like(mine, np.nan)
        d[m] = np.abs(mine - ref)[m]

        a2 = axs[2].imshow(d, cmap="magma")
        axs[2].set_title("abs diff (m)")
        plt.colorbar(a2, ax=axs[2])

        plt.tight_layout()

        p = os.path.join(out_dir, f"cmp_{f:05d}.png")
        plt.savefig(p, dpi=100)
        plt.close()

        print(f"saved {p}")


if __name__ == "__main__":
    main()