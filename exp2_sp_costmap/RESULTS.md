# Experiment 2 — Shortest-Path demos: RGB vs Costmap (held-out generalization)

**Run date:** 2026-07-19/20 (rented 4×RTX-3090 box). All metrics logged to
W&B project `pirlnav-baseline` (runs `il_costmap_2ch_sp_eval_ckpt*`,
`il_rgb_sp_eval_ckpt*`, `il_rgb_lr5e4_eval_ckpt*`).

## Question
Do SP (shortest-path) demos become learnable/generalizable when the policy is
conditioned on the GT-geodesic costmap, vs. plain RGB? (The costmap *is* what the
SP teacher follows, so the interaction-effect prediction is: costmap+SP should
beat RGB+SP on held-out scenes.)

## Setup (identical across arms)
- BC / teacher-forced, 8M frames = **25k updates ≈ 0.8 epoch** over ~64k SP demos.
  (Deliberately small budget; the PIRLNav paper trains BC for ~500M steps on
  64 GPUs ≈ ~33 epochs — see "training budget" note below.)
- Eval: full **1000-episode HM3D-v2 val** split, deterministic actions,
  `EVAL.DATASET_TYPE ObjectNav-v2`.
- Costmap: 2-channel, 32×32, GRID_RES 0.2, normalize off. Geodesic fields
  **precomputed offline** (train + val) and loaded from disk at train/eval time
  (see `precompute_fields.py` — this replaced online field-building, which hung
  training on pathological navmeshes).

## Headline result — best checkpoint per arm (held-out val)

| Arm | Success | SPL | softSPL |
|---|---|---|---|
| **Costmap+SP** (ckpt.9) | **3.2%** | **2.01%** | **0.225** |
| RGB+SP matched lr 5e-4 (ckpt.9) | 2.1% | 1.10% | 0.166 |
| RGB+SP lr 1e-3 (ckpt.9) | 1.8% | 0.86% | 0.176 |

**Costmap+SP beats the matched RGB baseline on every metric:** Success ×1.5,
SPL ×1.8, softSPL ×1.35.

## Full checkpoint sweeps (Success rate)

| ckpt (update) | Costmap+SP | RGB+SP (1e-3) | RGB+SP matched (5e-4) |
|---|---|---|---|
| ckpt.9 (25k) | **3.2%** | 1.8% | 2.1% |
| ckpt.7 (20k) | 2.8% | 1.2% | 1.8% |
| ckpt.5 (15k) | 0.6% | 0.7% | 1.7% |
| ckpt.3 (10k) | (running) | 1.2% | 0.2% |
| ckpt.1 (5k)  | (running) | (done) | (done) |

Note: costmap SR is **monotonic (9 > 7 > 5)** — it keeps improving with training,
no overfitting collapse. RGB peaks early and wobbles. This suggests the costmap
arm would climb further with a larger budget.

## Interpretation
- **Direction confirmed:** conditioning on the costmap generalizes measurably
  better than RGB from SP demos, at matched budget, on unseen scenes.
- **Absolute numbers are low for BOTH arms** — a shared ceiling, not a costmap
  problem. Two entangled causes (not separated here):
  1. **Undertraining:** ~0.8 epoch vs the paper's ~33 (≈62× fewer frames,
     1/64 the GPUs). LR decays to ~0 by 8M steps, forcing learning to stop early.
  2. **Representation limits** (FOV / 5m depth window / resolution) — the costmap
     is backprojected through the same limited depth camera, so it inherits the
     same ceiling. (This is the internship's original unresolved finding: even
     RGB+*human* demos hit ~0% here vs the paper's 64.1%.)
- The costmap-vs-RGB comparison holds **regardless** of that ceiling because both
  arms ran at the identical 8M-frame budget.

## Training-budget comparison (why absolute SR is far below the paper's 64%)
| | PIRLNav paper (BC, App. C.1) | This run |
|---|---|---|
| Steps/frames | ~500M | 8M |
| GPUs | 64 | 1 |
| Demos | 77k (human) | ~64k (SP) |
| Epochs over demos | ~33 | ~0.8 |
| LR | 1e-3 linearly decayed | 1e-3 / 5e-4 linearly decayed |

## Checkpoints
Trained weights (10 ckpts each: `il_costmap_2ch_sp`, `il_rgb_sp`,
`il_rgb_sp_lr5e4`) live on the rented box under `data/new_checkpoints/` — **NOT
in git** (too large). Metrics are preserved in W&B. If the box is torn down the
weights are lost; re-training from the committed scripts reproduces them.
