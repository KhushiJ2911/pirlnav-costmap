# Held-out eval — full checkpoint sweep

Full 1000-episode HM3D-v2 val, deterministic actions, matched 8M-frame budget (~0.8 epoch).
Checkpoint index maps to updates as ckpt.9 = 25k, ckpt.7 ≈ 20k, ckpt.5 ≈ 15k, ckpt.3 ≈ 10k, ckpt.1 ≈ 5k.
Numbers exported from Weights & Biases before pruning noise runs (run IDs kept for provenance).

## Phase 1 · Human · RGB — train `5evr8nyg` (il_rgb_1day)

| Checkpoint | Success | SPL | softSPL | W&B run |
|---|---|---|---|---|
| ckpt.3 (report) | 0.0% | 0.00% | 0.129 | `wrj46kdj` |

## Phase 1 · Human · Costmap 2-ch — train `wmxjgj7b` (il_costmap_2ch)

| Checkpoint | Success | SPL | softSPL | W&B run |
|---|---|---|---|---|
| ckpt.9 (report) | 0.0% | 0.00% | 0.108 | `cm2cheval1` |

## Phase 2 · SP · RGB (lr 1e-3) — train `ur9r5gxc` (il_rgb_sp)

| Checkpoint | Success | SPL | softSPL | W&B run |
|---|---|---|---|---|
| ckpt.1 | 0.0% | 0.00% | 0.044 | `47f6p5uc` |
| ckpt.3 | 1.2% | 0.58% | 0.165 | `i9e2wjkg` |
| ckpt.5 | 0.7% | 0.37% | 0.154 | `79p587yb` |
| ckpt.7 | 1.2% | 0.52% | 0.168 | `lpyi4qsw` |
| ckpt.9 (report) | 1.8% | 0.86% | 0.176 | `7j0451xk` |

## Phase 2 · SP · RGB (matched 5e-4) — train `8z0kcyer` (il_rgb_sp_lr5e4)

| Checkpoint | Success | SPL | softSPL | W&B run |
|---|---|---|---|---|
| ckpt.1 | 0.0% | 0.00% | 0.063 | `1raof1tl` |
| ckpt.3 | 0.2% | 0.08% | 0.124 | `vu2hah62` |
| ckpt.5 | 1.7% | 0.93% | 0.153 | `18zuxd61` |
| ckpt.7 | 1.8% | 0.89% | 0.152 | `qrglxarr` |
| ckpt.9 (report) | 2.1% | 1.10% | 0.166 | `plw47mrt` |

## Phase 2 · SP · Costmap 2-ch — train `n9rg9oef` (il_costmap_2ch_sp)

| Checkpoint | Success | SPL | softSPL | W&B run |
|---|---|---|---|---|
| ckpt.5 | 0.6% | 0.34% | 0.191 | `5bw5j6eh` |
| ckpt.7 | 2.8% | 1.68% | 0.225 | `bykjnjr4` |
| ckpt.9 (report) | 3.2% | 2.01% | 0.225 | `whz9w07i` |

## De-risk / overfit sanity check (SP demos) — IN-DISTRIBUTION, not generalization

Run before the full SP training to confirm the costmap arm could learn shortest-path demos
at all. The policy is overfit on a tiny scene set and evaluated on the **same scenes it trained
on** (`overfit_tiny/val` is a byte copy of `overfit_tiny/train`; `run_eval_overfit.sh` evals
"the SAME scene/episodes it memorized"). Two variants, A and B. **This is a sanity check, not a
generalization result** — the real held-out costmap number is 3.2% (above).

| Variant | Input | Success | SPL | softSPL | W&B run |
|---|---|---|---|---|---|
| A | RGB | 6.06% | 4.89% | 0.252 | `rgb_sp_A_eval` |
| A | Costmap | 75.0% | 70.97% | 0.827 | `cm_sp_A_eval` |
| B | RGB | 0.0% | 0.00% | 0.382 | `rgb_sp_B_eval` |
| B | Costmap | 100% | 93.88% | 0.928 | `cm_sp_B_eval` |

Note: the matching overfit *training* runs (`rgb_sp_overfit`, `cm_sp_overfit`) both show ~100%
"train success", but that metric is teacher-forced and not meaningful; the free-rollout eval
above is the real signal. These W&B runs were deleted in the 2026-07-21 cleanup; numbers
recovered from the pre-deletion run dump.
