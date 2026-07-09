# PIRLNav + WayPixel Costmap — Internship Handover

**Khushi Jaiswal · July 2026 · final day 2026-07-09**
Server: Phoenix (3× RTX A4000 16GB, shared). All W&B links live in project
[`pirlnav-baseline`](https://wandb.ai/jaiswalkhushi2911-coep-technological-university-/pirlnav-baseline).

## 0. One-paragraph summary

The PIRLNav BC pipeline was fully reproduced, certified end-to-end, and instrumented
(grad accumulation for matched effective batch 16,320 ≈ paper's 16,384). A costmap-
conditioned variant (per-pixel GT geodesic-to-goal, MasterNav/WayPixel style) was
built, debugged (two real fixes: field-build 121.8s→7s; 2-channel input transform),
and trained at a matched 8M-frame budget against an RGB baseline. **Behavioral result
at this budget: both arms score SR 0.000 on full 1000-ep HM3D-v2 val (softSPL: RGB
0.13, costmap 0.11)** — the oracle input does not convert to navigation under
imitation of *wandering human demos*. Leading hypothesis: the teacher, not the
interface, is the bottleneck (perfect imitation of these demos yields only SPL 0.47).
Next critical experiments: scripted greedy-descent probe (input-sufficiency test,
zero training) and shortest-path-demo BC (B′). Dustin's repo (offline zarr BC,
34.6 SR anchor) is the scaling path; it already has costmap channel plumbing.

## 1. Verified facts (each with evidence)

| Fact | Evidence |
|---|---|
| Training pipeline correct | overfit 1 scene → loss 0.000, acc 1.000 (W&B `il_overfit_long_2000`, `cm_overfit_1scene`) |
| Eval harness correct | same-scene eval through full eval path → SR 1.0000 (`overfit_eval_same_scene`) |
| Grad accum correct | unit test: optimizer steps exactly every N updates, params frozen between |
| Old runs' plateau cause | effective batch 128 vs 16,384 (Dustin's diagnosis, confirmed); NOT a bug |
| bnikm4aq "crash" | manual stop after flat loss; LR had re-warmed via patch_resume NUM_UPDATES change (avoid resume-with-changed-config) |
| Official PIRLNav ckpts | **gone from the internet** (S3 bucket deleted; issues #10/#11; no mirror/wayback). Encoder mirror: huggingface.co/gunjan050/ZSON |
| Paper numbers (verified in paper) | BC-77k: **64.1 SR / 27.1 SPL**; BC-20k: 52.0/20.6; BC+RL: 70.4/34.1. (The "54.5/38.6" in older notes is WRONG — not in the paper) |
| Dustin's repro anchor | 64 envs, batch 16k, 10% demo subset → **34.6 SR / 14.5 SPL** (checkpoint promised, not yet received) |
| Train/val category maps | identical (chair:0…sofa:5), ruled out as failure cause |
| HM3D-v2 val | 1000 episodes (val/content/); ALWAYS eval full split (TEST_EPISODE_COUNT -1) |

## 2. Experiments run (all on W&B)

| Run | Config | Result |
|---|---|---|
| `il_rgb_1day_10env_acc51` (5evr8nyg) | RGB BC, 10env×32steps×accum51 (batch 16,320), 25k updates = 8M frames ≈ 0.8 epoch, OVRL encoder | loss 1.79→**0.559** (flat final 30%), acc 0.744; matches Dustin's curve at matched steps (0.90 @ ~90 opt steps) |
| `il_rgb_1day_eval` (wrj46kdj) | full-val eval, ckpts 0/1/2 (ckpt.3–9 pending; run_eval_backlog.sh resumes) | SR 0.000 all; softSPL 0.010 (init) → 0.049 (0.8M) → **0.130** (1.6M) |
| `il_costmap_1day_10env_acc51` | 1-channel costmap (/30 global scale), same recipe, no pretrained encoder | **STALLED** at loss ~1.0 = action-prior floor for 1.3M+ frames; entropy rising → policy ignored input. Kept as ablation |
| `cm_overfit_1scene` | costmap overfit, 1 scene | loss 0.000, acc 1.000 → machinery fine, representation was the issue |
| `il_costmap_2ch_10env_acc51` | **2-channel transform** (see §3), same 8M budget | loss 1.79→**~0.63 still descending**, acc ~0.72 — fix worked at train level |
| `il_costmap_2ch_eval` (cm2cheval1) | full-val eval ckpts 1/3/5/7/9 | **SR 0.000 all**; softSPL 0.073/0.108/0.106 (ckpt 3/5/7); dist-to-goal ~5.4m. ckpt.9 finishing 2026-07-09 ~17:45 — check W&B |

## 3. The two real costmap fixes (in this codebase)

1. **Field build 121.8s → 7.0s**: `MultiGoalShortestPath` (one query per grid point,
   min over goals internally) instead of per-goal ShortestPath loops. Plus LRU cache
   keyed `(scene_id, object_category)` — ≤480 distinct fields exist (80 scenes × 6 cats).
   Plus compute at OUT 64×64 (downsample depth first; pixel-count dominates cost) and
   cached ray-direction maps. Result: training fps 25→135. `pirlnav/task/costmap_sensor.py`.
2. **2-channel input transform** (`pirlnav/policy/costmap_policy.py::CostmapTransform`):
   ch0 = absolute cost/30 (STOP cue), ch1 = per-frame min-max stretched (direction
   gradient at full contrast). With global /30 alone the within-frame gradient spans
   ~5% of input range (measured: costs 5.6–7.0 → 0.19–0.23) and a from-scratch encoder
   converges to the action-prior floor while ignoring the input (the stalled run).

## 4. Central open finding + next experiments (in priority order)

**Finding:** GT-costmap BC ≈ RGB BC behaviorally (both SR 0) at 8M frames on human
demos. Interpretation: human demos *wander* (perfect imitation = SPL 0.47, measured);
imitating explorers teaches exploring, so the oracle input cannot express itself
through this objective; STOP (1-in-~130 actions) is unlearned at <1 epoch.

1. **Scripted greedy-descent probe (NOT YET RUN, ~50 lines, eval-only):** turn toward
   min visible cost, forward, STOP when near-zero cost visible. High SR → input
   sufficient, teacher is bottleneck → do (2). Low SR → representation insufficient
   (FOV/5m depth window/64×64) → fix input first. This splits the world cleanly.
2. **B′ — shortest-path demos:** regenerate demos with habitat's ShortestPathFollower
   (SPL-optimal teacher, matched to what the costmap knows). Prediction: costmap arm
   improves dramatically; RGB arm may too (worth running both = 2×1 GPU-day).
3. **Scale via Dustin's repo** (github.com/DustinCraggs/pirlnav, `khushi` branch):
   offline zarr BC, shuffled batches (better diversity than 512 envs), cheap epochs.
   His 10% subset index is committed in-repo (`data/zarr/ten_percent/split_0/`).
   Costmap plumbing already exists (`POLICY.RGB_ENCODER.costmap_channels`,
   `costmap_names: [gt_costmap|goal_costmap|predicted_costmap]`) but the costmap
   *generators* are NOT in the repo (live in his private sg_habitat) — port
   `costmap_sensor.py` logic as a generator class in `gen_representation_dataset.py`.
   NOTE his stack: py3.10, habitat-sim from source, but **torch 1.12.1+cu113 — will
   NOT run on H100 (sm_90)**; needs torch/CUDA bump + habitat-sim rebuild.
4. **Anchor eval:** when Dustin sends his 34.6 checkpoint, eval on this harness
   (run_eval.sh) to certify absolute numbers.

## 5. Codebase inventory (this repo, branch `costmap-intern`)

- `pirlnav/algos/agent.py` — grad accumulation + bc_metrics (accuracy/top-k)
- `pirlnav/il_trainer.py` — accum cadence; EVAL.DATASET_TYPE configurable (hardcoded
  ObjectNav-v1 broke same-scene eval on HD-format episodes)
- `pirlnav/config.py` — NUM_ACCUM_STEPS, COSTMAP_SENSOR node, COSTMAP_ENCODER node, EVAL.DATASET_TYPE
- `pirlnav/task/costmap_sensor.py` — online geodesic costmap sensor (validated vs
  offline GT: MAE 0.06–0.10m). §3 optimizations included
- `pirlnav/policy/costmap_policy.py` — ObjectNavILCostmapPolicy + 2-channel transform
- `configs/tasks/objectnav_hm3d_costmap.yaml`, `configs/experiments/il_objectnav_costmap.yaml`
- `habitat_lab_eval_fix.patch` — **required for eval**: rnn_state_encoder contiguous
  fix (submodule change; apply with `git -C habitat-lab apply ../habitat_lab_eval_fix.patch`)
- `run_train_1gpu.sh` / `run_train_costmap_1gpu.sh` — certified 1-GPU configs (16GB:
  10 envs max, 12 OOMs; num_mini_batch ≤ num_envs — generator splits by env)
- `run_eval*.sh` — full-val eval; `run_eval_backlog.sh` (RGB ckpts 3–9, resumes W&B
  run wrj46kdj); `run_eval_costmap.sh`; `run_verify.sh`/overfit scripts (pipeline certification)
- `scripts_debug_costmap.py` — dumps live costmap observations + stats (the tool that
  found the contrast bug)
- `costmap_gen/*.py` — Aditya's offline GT reference (validation only, not training path)
- `EXPERIMENT_PLAN.md` — claim structure agreed direction; `logs/` — eval results text logs
- `patch_*.py` — historical one-shot patches (already applied; keep for provenance)

## 6. Data & artifacts (Phoenix paths — will be lost with account; copy what matters)

- Demos: `data/datasets/objectnav/objectnav_hm3d_hd/` (HF: axel81/pirlnav)
- Val: `data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2/val/` (1000 eps)
- Scenes: symlinks → `/scratch/toponavgroup/indoor-topo-loc/datasets/hm3d_navigation/hm3d_v0.2`
- Encoder: `data/visual_encoders/omnidata_DINO_02.pth` (HF: gunjan050/ZSON)
- **Checkpoints (NOT in git, ~200MB each):** `data/new_checkpoints/{il_1gpu_10env,
  il_costmap_2ch}/ckpt.9.pth` are the two final models — upload to HF/Drive if wanted;
  all metrics survive on W&B regardless
- Env recipe: conda `pirlnav` (py3.7, torch 1.12.1+cu113, habitat-sim 0.2.2, lmdb via
  conda-forge, webdataset 0.1.40); W&B key via `WANDB_API_KEY` env (new-format keys
  break `wandb login`)

## 7. Known traps (learned the hard way)

- Resume loads config FROM the resume state: changing NUM_UPDATES on resume re-warms
  the LR (bnikm4aq). Set budgets up front; never extend via resume.
- `metrics/success≈1` during BC training is teacher-forced replay — meaningless.
  Only `losses/*` are training signals; behavior lives in eval only (RGB: acc 0.744
  yet SR 0.000 — compounding errors are real).
- tqdm-style eval: undertrained policies never STOP → every episode runs the 500-step
  cap → eval is slow (~10–16 s/ep) and early "0/1000" stretches are normal.
- Eval NUM_ENVIRONMENTS is pure parallelism (episodes independent) — not a
  train/eval mismatch. CPU contention, not VRAM, caps it on shared boxes.
- habitat ddp_utils reads MAIN_PORT (not MASTER_PORT); default 8738 collides across
  concurrent runs.
- Non-interactive shells skip ~/.bashrc: scripts must self-source the W&B key + conda.
