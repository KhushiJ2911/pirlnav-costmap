"""Experiment 2 / B' (HANDOVER §4): regenerate demonstrations with an SPL-optimal
teacher (habitat's ShortestPathFollower) in the exact objectnav_hm3d_hd format, so
DEMONSTRATION_SENSOR / InflectionWeightSensor / the whole BC pipeline replay them
unchanged — only the DATA_PATH changes in the training scripts.

Per episode: pick the goal view-point with minimum geodesic distance from the
start, follow the shortest path to it, record the action sequence, verify success
through the task's own metrics, and rewrite the episode's reference_replay
(index 0 stays a placeholder entry — DemonstrationSensor reads from index 1).
Episodes whose goal is unreachable or whose rollout fails are dropped by default.

Usage (on the GPU box, from ~/pirlnav; shard by scene across CPU processes):
  python generate_sp_demos.py --num-shards 8 --shard-index 0 \
      --src data/datasets/objectnav/objectnav_hm3d_hd \
      --dst data/datasets/objectnav/objectnav_hm3d_sp_gen
  # after all shards finish, it prints the suggested INFLECTION_COEF; recompute
  # the global one over all shards with:
  python generate_sp_demos.py --report-coef data/datasets/objectnav/objectnav_hm3d_sp_gen
"""

import argparse
import glob
import gzip
import json
import os
import shutil
import time


def _poskey(pos):
    # Join key: habitat re-indexes episode_id on load (MTurk ids -> "0","1",...),
    # so match generated replays to raw episodes by start_position instead.
    return tuple(round(float(x), 3) for x in pos)


def scene_files(src, split="train"):
    files = sorted(glob.glob(os.path.join(src, split, "content", "*.json.gz")))
    if not files:
        raise FileNotFoundError(f"no content files under {src}/{split}/content/")
    return files


def replay_stats(episodes):
    """(total_steps, inflections) over reference_replay action sequences,
    matching InflectionWeightSensor's definition (timestep 1..len-1, weight at
    t where action[t-1] != action[t])."""
    total, inflections = 0, 0
    for ep in episodes:
        rr = [e["action"] for e in ep["reference_replay"]]
        for t in range(1, len(rr)):
            total += 1
            if rr[t - 1] != rr[t]:
                inflections += 1
    return total, inflections


def report_coef(dst, split="train"):
    total, infl = 0, 0
    for f in scene_files(dst, split):
        with gzip.open(f, "rt") as fh:
            d = json.load(fh)
        t, i = replay_stats(d["episodes"])
        total += t
        infl += i
    coef = total / max(infl, 1)
    print(f"total steps {total}, inflections {infl}")
    print(f"suggested TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF = {coef}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/datasets/objectnav/objectnav_hm3d_hd")
    parser.add_argument("--dst", default="data/datasets/objectnav/objectnav_hm3d_sp_gen")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--goal-radius", type=float, default=0.1)
    parser.add_argument("--max-episodes-per-scene", type=int, default=-1,
                        help="cap for smoke tests; -1 = all")
    parser.add_argument("--keep-failures", action="store_true",
                        help="keep episodes whose SP rollout did not succeed")
    parser.add_argument("--report-coef", metavar="DST", default=None,
                        help="only recompute INFLECTION_COEF over generated data")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="pirlnav-baseline")
    args = parser.parse_args()

    if args.report_coef:
        report_coef(args.report_coef, args.split)
        return

    wb = None
    if not args.no_wandb:
        try:
            import wandb as wb

            wb.init(
                project=args.wandb_project,
                name=f"gen_sp_demos_s{args.shard_index}of{args.num_shards}",
                config=dict(
                    src=args.src,
                    dst=args.dst,
                    goal_radius=args.goal_radius,
                    keep_failures=args.keep_failures,
                    num_shards=args.num_shards,
                    shard_index=args.shard_index,
                ),
            )
        except Exception as e:
            print(f"WARNING: wandb logging disabled ({e})")
            wb = None

    import habitat
    import numpy as np
    import pirlnav  # noqa: F401  (registers ObjectNav-v2 dataset/task)
    from habitat.sims.habitat_simulator.actions import HabitatSimActions
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
    from pirlnav.config import get_task_config

    action_names = {
        HabitatSimActions.STOP: "STOP",
        HabitatSimActions.MOVE_FORWARD: "MOVE_FORWARD",
        HabitatSimActions.TURN_LEFT: "TURN_LEFT",
        HabitatSimActions.TURN_RIGHT: "TURN_RIGHT",
    }

    files = scene_files(args.src, args.split)[args.shard_index :: args.num_shards]
    print(f"shard {args.shard_index}/{args.num_shards}: {len(files)} scenes")

    # Top-level split file: copy once (holds category maps; content/ holds episodes).
    if args.shard_index == 0:
        top_src = os.path.join(args.src, args.split, f"{args.split}.json.gz")
        top_dst = os.path.join(args.dst, args.split, f"{args.split}.json.gz")
        os.makedirs(os.path.dirname(top_dst), exist_ok=True)
        shutil.copyfile(top_src, top_dst)
        print(f"copied {top_src} -> {top_dst}")

    config = get_task_config("configs/tasks/objectnav_hm3d.yaml")
    config.defrost()
    config.DATASET.TYPE = "ObjectNav-v2"
    config.DATASET.SPLIT = args.split
    config.DATASET.DATA_PATH = os.path.join(
        args.src, "{split}", "{split}.json.gz"
    )
    # Generation needs no visuals — a tiny depth sensor keeps the sim happy and cheap.
    config.SIMULATOR.AGENT_0.SENSORS = ["DEPTH_SENSOR"]
    config.SIMULATOR.DEPTH_SENSOR.WIDTH = 64
    config.SIMULATOR.DEPTH_SENSOR.HEIGHT = 64
    config.TASK.SENSORS = []
    config.TASK.MEASUREMENTS = ["DISTANCE_TO_GOAL", "SUCCESS", "SPL", "SOFT_SPL"]
    config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.ENVIRONMENT.ITERATOR_OPTIONS.CYCLE = False
    config.freeze()

    out_content = os.path.join(args.dst, args.split, "content")
    os.makedirs(out_content, exist_ok=True)

    grand_total, grand_infl, grand_kept, grand_dropped = 0, 0, 0, 0

    for fpath in files:
        scene = os.path.basename(fpath)[: -len(".json.gz")]
        out_path = os.path.join(out_content, f"{scene}.json.gz")
        if os.path.exists(out_path):
            print(f"[{scene}] output exists, skipping")
            continue

        with gzip.open(fpath, "rt") as fh:
            raw = json.load(fh)

        scene_cfg = config.clone()
        scene_cfg.defrost()
        scene_cfg.DATASET.CONTENT_SCENES = [scene]
        scene_cfg.freeze()

        replays = {}  # episode_id -> (list of action-name, success)
        env = habitat.Env(config=scene_cfg)
        try:
            follower = ShortestPathFollower(
                env.sim, goal_radius=args.goal_radius, return_one_hot=False
            )
            n = len(env.episodes)
            if args.max_episodes_per_scene > 0:
                n = min(n, args.max_episodes_per_scene)
            t0 = time.time()
            for i in range(n):
                env.reset()
                ep = env.current_episode
                start = np.asarray(env.sim.get_agent_state().position)

                # Nearest goal view-point by geodesic distance.
                best_vp, best_d = None, float("inf")
                for goal in ep.goals:
                    vps = getattr(goal, "view_points", None) or []
                    cands = [vp.agent_state.position for vp in vps] or [goal.position]
                    for p in cands:
                        d = env.sim.geodesic_distance(start, p)
                        if d < best_d:
                            best_d, best_vp = d, np.asarray(p)

                if best_vp is None or not np.isfinite(best_d):
                    replays[_poskey(ep.start_position)] = (None, False)
                    continue

                actions = []
                while not env.episode_over:
                    a = follower.get_next_action(best_vp)
                    if a is None:
                        a = HabitatSimActions.STOP
                    a = int(a)
                    actions.append(action_names.get(a, "STOP"))
                    env.step(a)
                success = bool(env.get_metrics().get("success", 0.0) > 0)
                replays[_poskey(ep.start_position)] = (actions, success)
                if (i + 1) % 50 == 0 or i + 1 == n:
                    ok = sum(1 for a, s in replays.values() if s)
                    print(
                        f"[{scene}] {i+1}/{n} sp-success {ok}/{len(replays)} "
                        f"({(i+1)/(time.time()-t0):.2f} ep/s)",
                        flush=True,
                    )
        finally:
            env.close()

        kept = []
        for ep in raw["episodes"]:
            actions, success = replays.get(_poskey(ep["start_position"]), (None, False))
            if actions is None or (not success and not args.keep_failures):
                grand_dropped += 1
                continue
            # Index 0 is a placeholder (DemonstrationSensor reads from index 1).
            ep["reference_replay"] = [{"action": "STOP"}] + [
                {"action": a} for a in actions
            ]
            ep["attempts"] = 1
            kept.append(ep)
        raw["episodes"] = kept
        grand_kept += len(kept)

        with gzip.open(out_path, "wt") as fh:
            json.dump(raw, fh)
        t, i_ = replay_stats(kept)
        grand_total += t
        grand_infl += i_
        print(f"[{scene}] wrote {len(kept)} episodes -> {out_path}")
        if wb is not None:
            wb.log(
                dict(
                    scenes_done=wb.run.step + 1 if wb.run.step else 1,
                    kept_total=grand_kept,
                    dropped_total=grand_dropped,
                    steps_total=grand_total,
                    inflections_total=grand_infl,
                )
            )

    print(
        f"DONE shard {args.shard_index}: kept {grand_kept}, dropped {grand_dropped}, "
        f"steps {grand_total}, inflections {grand_infl}, "
        f"shard-local coef {grand_total / max(grand_infl, 1):.6f}"
    )
    print("After ALL shards finish, run --report-coef for the global coefficient.")
    if wb is not None:
        wb.summary.update(
            dict(
                kept=grand_kept,
                dropped=grand_dropped,
                steps=grand_total,
                inflections=grand_infl,
                shard_local_coef=grand_total / max(grand_infl, 1),
            )
        )
        wb.finish()


if __name__ == "__main__":
    main()
