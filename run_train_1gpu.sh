#!/bin/bash
# Certified 1-GPU (A4000 16GB) config, probed 2026-07-02:
#   10 envs x 32 steps x accum 51 = effective batch 16320 (~= paper/Dustin's 16384)
#   num_mini_batch 5 -> 64-frame backward chunks: peak 12.4GB (75%), ~93 fps
#   12 envs OOMs; 8 envs @ 64 steps peaks at 98% (too tight for multi-day runs).
# NOTE num_steps 32 (not 64): halves VRAM; BPTT window 32. Batch size compensated via accum.
# ~93 fps => ~8M frames/day => ~490 optimizer steps/day.
# 1-DAY BUDGET: NUM_UPDATES 25000 (~= 93fps * 86400s / 320 frames-per-update) so the
# linear LR decay completes over the actual run; 10 ckpts = SR-vs-frames curve point every ~2.4h.
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8981
export MAIN_PORT=8981  # habitat ddp_utils reads MAIN_PORT (default 8738 collides with concurrent runs)
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
eval "$(grep 'WANDB_API_KEY' ~/.bashrc)"
# self-contained env activation (tmux/cron shells start in base)
source ~/miniconda/etc/profile.d/conda.sh && conda activate pirlnav
cd ~/pirlnav

python -u -m run \
    --exp-config configs/experiments/il_objectnav.yaml \
    --run-type train \
    TENSORBOARD_DIR tb/il_1gpu_10env \
    CHECKPOINT_FOLDER data/new_checkpoints/il_1gpu_10env \
    NUM_UPDATES 25000 \
    NUM_ENVIRONMENTS 10 \
    NUM_ACCUM_STEPS 51 \
    IL.BehaviorCloning.num_steps 32 \
    IL.BehaviorCloning.num_mini_batch 5 \
    NUM_CHECKPOINTS 10 \
    LOG_INTERVAL 50 \
    WRITER_TYPE wb \
    WB.PROJECT_NAME pirlnav-baseline \
    WB.RUN_NAME il_rgb_1day_10env_acc51 \
    POLICY.RGB_ENCODER.pretrained_encoder data/visual_encoders/omnidata_DINO_02.pth \
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS 10000 \
    TASK_CONFIG.DATASET.DATA_PATH "data/datasets/objectnav/objectnav_hm3d_hd/{split}/{split}.json.gz" \
    TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.234951275740812
