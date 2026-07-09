# Experiment Plan: Geodesic Costmaps as an Intermediate Representation for ObjectNav

**Khushi Jaiswal — draft for advisor/senior sign-off, July 2026**

## Research question

How much of ObjectNav's difficulty is perception/planning vs control — and where does
learning effort (BC, RL) actually pay off when navigation is factored through a
geodesic costmap (MasterNav/plann3r interface), instead of learned end-to-end (PIRLNav)?

**Hypothesis.** The costmap carries the hard part. A costmap-conditioned controller
should approach its ceiling with orders of magnitude less compute than end-to-end RGB;
the interesting science is (a) what breaks when the GT costmap is replaced by a
predicted one, and (b) which method best closes that gap — including whether RL
finetuning (my original task) helps the controller under imperfect maps, rather than
on the GT oracle where it has no headroom.

## Status (done)

- PIRLNav IL pipeline fully certified end-to-end: overfit test memorizes (action
  accuracy 1.0) and same-scene eval through the full eval harness gives SR 1.0 —
  no bugs; earlier ~0% val SR was training scale (~100 optimizer steps vs paper's ~80k).
- Gradient accumulation implemented + unit-tested; effective batch matched to 16,384.
- Certified 1-GPU config (A4000 16GB): 10 envs × 32 steps × accum 51, peak 12.4GB,
  ~93 fps (~8M frames/day). Eval protocol fixed to full 1000-episode HM3D val.
- Online CostmapSensor written and validated against offline GT (MAE 0.06–0.10 m,
  ~20 ms/frame); not yet wired into training.
- Note: official PIRLNav checkpoints are no longer downloadable (S3 bucket deleted,
  see repo issues #10/#11) — requesting a known-good checkpoint to anchor our eval.

## Experiments

**Budget note:** compute is constrained to ~1 GPU-day per training run. The design is
therefore a matched-budget sample-efficiency study: both arms get an identical 1-day /
~8M-frame budget (certified config: 10 envs × 32 steps × accum 51, effective batch
16320 ≈ paper's 16384). Published large-budget RGB anchors — cited, not retrained:
BC on 77k demos = 64.1 SR / 27.1 SPL; BC on 20k demos = 52.0 / 20.6 (PIRLNav paper);
Dustin's reproduction (64 envs, 16k batch, 10% demo subset) = 34.6 / 14.5. Headline
figure: SR/loss vs frames for both arms over the same budget.

| # | Experiment | Question it answers | Est. cost (1× A4000) |
|---|---|---|---|
| A | RGB PIRLNav BC, 1-day / ~8M-frame budget | end-to-end performance per unit compute (large-scale point cited from paper) | 1 GPU-day |
| B | BC on **GT** costmap, identical budget/protocol | is control the easy part? sample-efficiency gap vs A | 1 GPU-day |
| C | B's controller evaluated on noisy / degraded costmaps | robustness gap: what breaks when the oracle becomes a prediction | eval-only (~hours) |
| C′ (if time) | One gap-closing method (noise-injection BC or RL finetune) under imperfect maps | where RL belongs in the modular setting (repositioned original task) | ~1 GPU-day |
| B′/D (future work) | Shortest-path demos; task-aware predictor training (distill-through-controller / consistency vs per-pixel MSE) | right teacher for modular agent; can downstream signal improve costmaps? | — |

**Minimum publishable unit:** A + B + C — a sample-efficiency and robustness analysis
of a geodesic-costmap interface vs end-to-end, at rigorously matched budget, with
"costmap reduces ObjectNav sample complexity by ~two orders of magnitude" as the
candidate headline claim (X% SR at a budget where RGB is ~0% and published RGB needs
~1.3B frames for 64.1 SR). C′ adds the RL answer if time allows. GT-only results are
explicitly framed as upper bounds (privileged input), never headline numbers.

## Asks

1. Sign-off on this claim structure before long GPU runs.
2. Dustin: your successful 2-GPU/64-env checkpoint, to anchor our full-val eval
   against a known-good number (official checkpoints are gone from the internet).
3. Confirm target venue/deadline so budget A is scoped accordingly (CoRL / ICRA / IROS?).
