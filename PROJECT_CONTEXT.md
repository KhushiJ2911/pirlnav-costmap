# Project Context — read this first (written 2026-07-17)

This file captures everything about the project's history and current state, so that
anyone (or any AI assistant) picking up this repo cold — e.g. on a rented GPU box with
no access to earlier chat sessions or the lab server — has the full picture. Companion
docs: `HANDOVER.md` (internship-end technical handover, 2026-07-09), `EXPERIMENT_PLAN.md`
(the experimental design), `RENTAL_SETUP.md` (environment/data setup runbook).

## Who / what / when

- Khushi Jaiswal (github `KhushiJ2911`, W&B `jaiswalkhushi2911`). Internship ended
  **2026-07-09**; lab server ("Phoenix", 3x RTX A4000 16GB) access is **gone**, along
  with everything symlinked there (datasets, conda env, checkpoints). Only this repo
  and the W&B project survive.
- Post-internship plan: continue the research on **rented GPUs (Lium, 4x RTX 3090
  24GB)** with the goal of (a) getting a result worth publishing/showcasing and
  (b) building a GitHub Pages internship writeup (advisor's spec: pick a template like
  ObjectReact's page, title + summary of internship work, visuals, link to public
  reproducible code with install/train/eval docs, references, no fluff, email the link).
  Decision made 2026-07-17: **hold the page until experiment results land; run
  HANDOVER §4 experiments 1 and 2 first.**

## The project in one paragraph

PIRLNav (CVPR 2023, arXiv 2301.07302) trains ObjectNav agents by behavior cloning on
77k human demos (64.1% SR) then RL-finetuning (70.4% SR). MASt3R-Nav
(https://mast3r-nav.github.io/) shows a dense per-pixel cost-to-goal representation
("WayPixel costmap") feeding a small controller beats end-to-end learning on its tasks.
The internship question: **if PIRLNav's BC policy is conditioned on an oracle (ground
truth) WayPixel-style costmap instead of RGB, does ObjectNav's difficulty collapse?**
Answer so far: **no — at a matched 8M-frame budget both arms score SR 0.000 on full
HM3D-v2 val** (RGB softSPL 0.130, costmap 0.108). Leading interpretation: the teacher
(wandering human demos, perfect imitation = SPL 0.47) is the bottleneck, not the input
interface. BC's objective punishes navigating better than the teacher, so an oracle
input cannot express itself.

## Verified experiment record (W&B project `pirlnav-baseline`, all public via API)

Entity: `jaiswalkhushi2911-coep-technological-university-`. Key runs (IDs verified
2026-07-17 via the W&B GraphQL API):

| Run ID | Name | What | Result |
|---|---|---|---|
| 5evr8nyg | il_rgb_1day_10env_acc51 | RGB BC, 8M frames | loss 1.79→0.559, acc 0.744, 154 fps |
| wrj46kdj | il_rgb_1day_eval | RGB full-val eval | **SR 0, SPL 0**, softSPL 0.129, dist-to-goal 5.37m |
| 2pxuefij | il_costmap_1day_10env_acc51 | 1-ch costmap (stalled ablation) | stopped at 2.1M frames, loss stuck ~0.85-1.0 (action-prior floor, input ignored) |
| kephhe5e / 4kcqjgao | cm_overfit_1scene | costmap pipeline certification | loss 0.000, acc 1.000 |
| wmxjgj7b | il_costmap_2ch_10env_acc51 | 2-ch costmap fix, 8M frames | loss 1.79→0.718, acc 0.744, 134 fps |
| cm2cheval1 | il_costmap_2ch_eval | costmap full-val eval | **SR 0, SPL 0**, softSPL 0.108, dist-to-goal 5.33m |

Earlier repro runs (il_ovrl_repro*, gradaccum tests, bnikm4aq crash-that-wasn't) are
documented in `HANDOVER.md` §1-2. NOTE: during BC training `metrics/success≈1` is
teacher-forced replay and meaningless; only `losses/*` matter at train time.

## How the code works — the non-obvious essentials

1. **Training is teacher forcing through the live simulator.** In
   `pirlnav/il_trainer.py` (`_compute_actions_and_step_envs`), the env is stepped with
   the *demo's* `next_actions` (from `DEMONSTRATION_SENSOR`), never the policy's. The
   policy only appears in the loss (`pirlnav/algos/agent.py`): inflection-weighted
   cross-entropy vs. the human action. This is why train-time success is meaningless
   and why 74% action accuracy coexists with 0% eval SR (compounding errors at eval,
   when the policy acts for itself for the first time).
2. **Grad accumulation** (`NUM_ACCUM_STEPS`, applied in `_update_agent`): 10 envs x 32
   steps x accum 51 = effective batch 16,320 ≈ paper's 16,384 on a single 16GB GPU.
   This fixed the earlier plateau (old runs used batch 128 — diagnosis by Dustin,
   confirmed by unit test; NOT a bug).
3. **CostmapSensor** (`pirlnav/task/costmap_sensor.py`): per (scene, category), builds
   a geodesic-distance field over the navmesh grid (0.2m res) using ONE
   `MultiGoalShortestPath` query per grid point (fix #1: 121.8s→7.0s per field) with an
   LRU cache (≤480 distinct fields exist); per frame, backprojects depth pixels to
   world points and KD-tree-looks-up their geodesic cost, output 64x64. Validated vs.
   offline GT (`costmap_gen/`): MAE 0.06-0.10m. Training fps 25→135.
4. **CostmapTransform** (`pirlnav/policy/costmap_policy.py`, fix #2): 1-channel global
   /30 scaling left within-frame gradients spanning ~4% of input range → from-scratch
   encoder ignored the input, policy converged to the action-prior floor. Fix: ch0 =
   absolute cost/30 (STOP cue), ch1 = per-frame min-max stretch (direction gradient at
   full contrast). Diagnosed with `scripts_debug_costmap.py`.
5. Costmap encoder trains **from scratch** (OVRL weights are RGB-only) — a known
   asymmetry favoring the RGB arm; disclose it when presenting results.

## The next experiments (decided; from HANDOVER §4)

**Experiment 1 — scripted greedy-descent probe (FIRST; not yet run; scripts READY).**
Eval-only, zero training: turn toward min visible cost, forward, STOP when near-zero
cost visible. High SR → input sufficient, teacher is the bottleneck → run exp 2. Low
SR → representation insufficient (FOV / 5m depth window / 64x64) → fix the input
before any training. Scripts (written 2026-07-17, logic unit-tested on CPU, not yet
run against the simulator): `probe_policy.py` (pure-numpy controller: greedy descent
+ scan-when-blind + collision sidestep + oscillation breaker; tests in
`test_probe_policy.py`), `run_probe_greedy.py` (habitat harness, scene-sharded,
`--merge` to combine shards), `run_probe.sh` (runs 4 shards + merged summary).
Design caveat: greedy descent is myopic (min-cost pixel can be across a wall); treat
"high SR" as the sufficiency signal, but a mid/low SR needs inspection before
concluding the input is insufficient.

**Experiment 2 — B′, shortest-path demos (conditional on exp 1 passing; scripts READY).**
Regenerate demos with habitat's `ShortestPathFollower` (SPL-optimal teacher), write
them in the same HD-demo format so `DEMONSTRATION_SENSOR` replays them unchanged, and
rerun BOTH arms (RGB + costmap) at the same matched budget. Scripts (written
2026-07-17, not yet run): `generate_sp_demos.py` + `run_gen_sp_demos.sh` (scene-sharded
generation into `data/datasets/objectnav/objectnav_hm3d_sp_gen/`, drops
unreachable/failed episodes, reports the recomputed INFLECTION_COEF), then
`run_train_1gpu_sp.sh` (RGB arm) and `run_train_costmap_1gpu_sp.sh` (costmap arm) —
both REQUIRE the `INFLECTION_COEF` env var from the generator's report. Eval with the
existing `run_eval.sh` / `run_eval_costmap.sh` pointed at the new checkpoint folders
(`il_rgb_sp`, `il_costmap_2ch_sp`). The sharp prediction (the
publishable interaction effect): PIRLNav's paper shows SP demos are a *terrible*
teacher for RGB (6.4% SR vs 64.1% on human demos — the policy can't see the privileged
map the SP teacher follows), but the costmap arm DOES see that map (the oracle geodesic
field is exactly what the SP teacher follows). So SP demos should stay bad for RGB but
become good for costmap. If it holds: "the costmap interface changes which teachers are
learnable" — a result in neither paper. Cost: 2x 1-GPU-day (parallelizable on 4x3090).

Experiments 3 (scale via Dustin's repo) and 4 (anchor eval vs his 34.6-SR checkpoint)
are later / blocked; see HANDOVER §4 and "Dustin's repo" below.

## Dustin's repo state (checked 2026-07-17)

`github.com/DustinCraggs/pirlnav`, branch `khushi` (public). Last commit `47f3808c`
(2026-07-08): "Fix issues with evals; use diverse scenes and goals in example 10-ep
index; add option to freeze and gradually warm up pre-trained encoder [untested]" —
this postdates Khushi's last chat with him (7/7) and touches eval correctness, so
re-verify his 34.6 anchor number against it. His stack: py3.10, torch 1.12.1+cu113
(fine on 3090/Ampere), offline zarr BC. Costmap config plumbing exists but is
**commented out** in `configs/tasks/objectnav_hm3d.yaml` (REPRESENTATION_GENERATOR
`ground_truth_costmap`, PVR `*costmap*` keys), and the costmap *generators* are NOT in
the repo (they live in his private `sg_habitat`). His 10% zarr split index is committed
at `data/zarr/ten_percent/split_0/`. Checkpoint (34.6 SR) promised but not received.

## Paper numbers to cite correctly (verified against the PIRLNav paper PDF)

- BC on 77k human demos: **64.1 SR / 27.1 SPL**; BC-20k: 52.0/20.6; BC→RL: **70.4/34.1**
  (val), 65.0/33.0 on test-standard. The "54.5/38.6" seen in older notes is WRONG.
- BC on 240k shortest-path demos: **6.4 SR / 5.0 SPL** (imitation gap — key for exp 2).
- BC on 70k frontier-exploration demos: 44.9/21.5.
- MASt3R-Nav (project page): PixelReact SPL 81.77 vs ObjectReact 51.5 (image-goal);
  4-task avg SPL 52.79 vs GNM 27.62. Its WayPixel costmap is built from MASt3R
  correspondences; ours substitutes the simulator's GT geodesic field (oracle bound).

## Known traps (full list in HANDOVER §7 — do not skip)

Resume re-warms LR if NUM_UPDATES changes (never extend via resume); MAIN_PORT
collisions between concurrent runs; non-interactive shells need self-sourced conda +
WANDB_API_KEY (legacy-format key only); eval of non-stopping policies runs the full
500-step cap (~10-16 s/ep — slow is normal); always eval the full 1000-ep val split
(TEST_EPISODE_COUNT -1); `habitat_lab_eval_fix.patch` must be applied or eval breaks.

## What exists locally vs. what's gone

In-repo and safe: all code, configs, run scripts, docs, eval text logs (`logs/*.log`),
costmap debug images (`logs/costmap_debug/*.png` — useful for the web page later).
GONE with Phoenix: `data/` (scenes, demos, val episodes, encoder), conda env, and the
two final checkpoints (`il_1gpu_10env/ckpt.9.pth`, `il_costmap_2ch/ckpt.9.pth`) unless
Khushi still has a copy — metrics survive on W&B regardless. Every dataset is
re-downloadable (verified 2026-07-17, exact URLs in `RENTAL_SETUP.md`) except HM3D
scenes, which need Khushi's own Matterport API token.
