"""Dump real costmap observations from the costmap task and check they carry signal.
Saves PNGs (costmap | depth side by side) to logs/costmap_debug/ and prints stats.
A healthy costmap: varies across pixels (gradient structure), changes as the agent
moves/turns, fraction-at-max_cost well below 1.0, values in [0, 30]."""
import os
import numpy as np
import cv2
import habitat
from pirlnav.config import get_config

OUT = "logs/costmap_debug"
os.makedirs(OUT, exist_ok=True)

cfg = get_config(
    "configs/experiments/il_objectnav_costmap.yaml",
    [
        "TASK_CONFIG.DATASET.DATA_PATH",
        "data/datasets/objectnav/overfit_tiny/{split}/{split}.json.gz",
        "TASK_CONFIG.DATASET.SPLIT",
        "train",
    ],
)

env = habitat.Env(config=cfg.TASK_CONFIG)
obs = env.reset()
print("obs keys:", sorted(obs.keys()))
print("episode object_category:", env.current_episode.object_category)

# forward + turns: the costmap must change with pose
actions = [1, 1, 2, 2, 1, 3, 3, 1, 1, 1]
MAX_COST = float(cfg.TASK_CONFIG.TASK.COSTMAP_SENSOR.MAX_COST)

for i, a in enumerate([None] + actions):
    if a is not None:
        obs = env.step(a)
    cm = np.asarray(obs["costmap"])[:, :, 0]
    dp = np.asarray(obs["depth"])[:, :, 0]
    at_max = float((cm >= MAX_COST - 1e-3).mean())
    print(
        f"step {i:2d} action={a}: costmap min={cm.min():6.2f} max={cm.max():6.2f} "
        f"mean={cm.mean():6.2f} std={cm.std():6.2f} at_max={at_max:5.1%} "
        f"unique={len(np.unique(cm))}  depth[min={dp.min():.2f} max={dp.max():.2f}]"
    )
    dp_small = cv2.resize(dp, (cm.shape[1], cm.shape[0]), interpolation=cv2.INTER_NEAREST)
    # per-frame contrast stretch for the costmap viz (absolute values printed above)
    lo, hi = cm.min(), max(cm.max(), cm.min() + 1e-6)
    viz = np.concatenate(
        [
            cv2.applyColorMap(((cm - lo) / (hi - lo) * 255).astype(np.uint8), cv2.COLORMAP_JET),
            cv2.applyColorMap((np.clip(dp_small, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS),
        ],
        axis=1,
    )
    cv2.imwrite(f"{OUT}/step{i:02d}_a{a}.png", cv2.resize(viz, (512, 256), interpolation=cv2.INTER_NEAREST))

env.close()
print(f"\nPNGs in {OUT}/ - costmap (jet: blue=near goal, red=far) | depth (viridis)")
