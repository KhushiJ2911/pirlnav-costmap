"""Experiment 1 (HANDOVER §4): scripted greedy-descent probe. Eval-only, zero training.

Drives GreedyCostmapPolicy (probe_policy.py) through the full HM3D-v2 val split
using the same CostmapSensor the learned policy sees. High SR => the costmap
input is sufficient and the teacher was the bottleneck (proceed to B' /
shortest-path demos). Low SR => representation insufficient (FOV / 5m depth
window / 64x64) — fix the input before more training.

Usage (on the GPU box, conda env active, from ~/pirlnav):
  python run_probe_greedy.py --out logs/probe_greedy.json
  # sharded across processes (by scene, round-robin):
  python run_probe_greedy.py --num-shards 4 --shard-index 0 --out logs/probe_s0.json
  # merge shard outputs:
  python run_probe_greedy.py --merge logs/probe_s*.json
"""

import argparse
import glob
import gzip
import json
import os
import time
from collections import defaultdict

import numpy as np

from probe_policy import GreedyCostmapPolicy


VAL_DATA_PATH = "data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2/{split}/{split}.json.gz"
TASK_CONFIG = "configs/tasks/objectnav_hm3d_costmap.yaml"


def list_val_scenes(split="val"):
    content_dir = os.path.join(
        os.path.dirname(VAL_DATA_PATH.format(split=split)), "content"
    )
    scenes = sorted(
        os.path.basename(p)[: -len(".json.gz")]
        for p in glob.glob(os.path.join(content_dir, "*.json.gz"))
    )
    if not scenes:
        raise FileNotFoundError(f"No scene files under {content_dir}")
    return scenes


def merge(paths):
    stats = defaultdict(list)
    n = 0
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        n += d["num_episodes"]
        for ep in d["episodes"]:
            for k, v in ep["metrics"].items():
                stats[k].append(v)
    print(f"episodes: {n}")
    summary = {"num_episodes": n}
    for k, v in sorted(stats.items()):
        print(f"  {k}: {np.mean(v):.4f}")
        summary[k] = float(np.mean(v))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="logs/probe_greedy.json")
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=-1,
                        help="cap for quick smoke tests; -1 = all")
    parser.add_argument("--stop-threshold", type=float, default=0.25)
    parser.add_argument("--merge", nargs="+", default=None,
                        help="merge shard output JSONs and exit (no simulator)")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="pirlnav-baseline")
    args = parser.parse_args()

    wb = None
    if not args.no_wandb:
        try:
            import wandb as wb

            wb.init(
                project=args.wandb_project,
                name=(
                    "probe_greedy_combined"
                    if args.merge
                    else f"probe_greedy_s{args.shard_index}of{args.num_shards}"
                ),
                config=dict(
                    stop_threshold=args.stop_threshold,
                    num_shards=args.num_shards,
                    shard_index=args.shard_index,
                    split=args.split,
                ),
            )
        except Exception as e:
            print(f"WARNING: wandb logging disabled ({e})")
            wb = None

    if args.merge:
        paths = []
        for pat in args.merge:
            paths.extend(glob.glob(pat))
        summary = merge(sorted(set(paths)))
        if wb is not None:
            wb.summary.update(summary)
            wb.finish()
        return

    # Habitat imports deferred so --merge works anywhere.
    import habitat
    import pirlnav  # noqa: F401  (registers CostmapSensor, ObjectNav-v2, etc.)
    from pirlnav.config import get_task_config

    scenes = list_val_scenes(args.split)
    shard_scenes = scenes[args.shard_index :: args.num_shards]
    print(f"shard {args.shard_index}/{args.num_shards}: {len(shard_scenes)} scenes")

    config = get_task_config(TASK_CONFIG)
    config.defrost()
    config.DATASET.TYPE = "ObjectNav-v1"  # standard val episodes (no replay data)
    config.DATASET.SPLIT = args.split
    config.DATASET.DATA_PATH = VAL_DATA_PATH
    config.DATASET.CONTENT_SCENES = shard_scenes
    config.freeze()

    policy = GreedyCostmapPolicy(
        max_cost=config.TASK.COSTMAP_SENSOR.MAX_COST,
        stop_threshold=args.stop_threshold,
    )

    results = []
    env = habitat.Env(config=config)
    try:
        total = len(env.episodes) if args.max_episodes < 0 else min(
            args.max_episodes, len(env.episodes)
        )
        t0 = time.time()
        for i in range(total):
            obs = env.reset()
            policy.reset()
            steps = 0
            while not env.episode_over:
                pos = env.sim.get_agent_state().position
                action = policy.act(obs["costmap"], position=pos)
                obs = env.step(action)
                steps += 1
            m = env.get_metrics()
            ep = env.current_episode
            results.append(
                dict(
                    episode_id=ep.episode_id,
                    scene_id=ep.scene_id,
                    object_category=getattr(ep, "object_category", None),
                    steps=steps,
                    metrics={
                        k: float(v)
                        for k, v in m.items()
                        if isinstance(v, (int, float, np.floating))
                    },
                )
            )
            if (i + 1) % 10 == 0 or i + 1 == total:
                sr = np.mean([r["metrics"].get("success", 0.0) for r in results])
                spl = np.mean([r["metrics"].get("spl", 0.0) for r in results])
                sspl = np.mean([r["metrics"].get("softspl", 0.0) for r in results])
                rate = (i + 1) / (time.time() - t0)
                print(
                    f"[{i+1}/{total}] SR {sr:.3f}  SPL {spl:.3f}  "
                    f"softSPL {sspl:.3f}  ({rate:.2f} ep/s)",
                    flush=True,
                )
                if wb is not None:
                    wb.log(
                        dict(
                            episodes_done=i + 1,
                            running_success=float(sr),
                            running_spl=float(spl),
                            running_softspl=float(sspl),
                            eps_per_sec=float(rate),
                        ),
                        step=i + 1,
                    )
    finally:
        env.close()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    agg = defaultdict(list)
    for r in results:
        for k, v in r["metrics"].items():
            agg[k].append(v)
    summary = {k: float(np.mean(v)) for k, v in agg.items()}
    with open(args.out, "w") as f:
        json.dump(
            dict(
                num_episodes=len(results),
                stop_threshold=args.stop_threshold,
                summary=summary,
                episodes=results,
            ),
            f,
            indent=1,
        )
    print("SUMMARY:", json.dumps(summary, indent=2))
    print("wrote", args.out)
    if wb is not None:
        wb.summary.update(dict(num_episodes=len(results), **summary))
        wb.finish()


if __name__ == "__main__":
    main()
