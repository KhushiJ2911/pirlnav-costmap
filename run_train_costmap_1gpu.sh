#!/bin/bash
# Costmap arm: identical recipe to the RGB baseline (run_train_1gpu.sh) with the
# input swapped: GT geodesic costmap (+depth sim sensor) instead of RGB.
# Matched on frames/batches/LR schedule: 10 envs x 32 steps x accum 51
# (effective batch 16320), NUM_UPDATES 25000 = 8M frames. At costmap fps
# (~40-70) this takes ~1.5-2 days wall-clock vs RGB's ~15h - that's fine,
# the comparison axis is frames, not hours.
# NOTE: no pretrained encoder - OVRL weights are 3-channel RGB and cannot init
# the 1-channel costmap encoder. Known asymmetry favoring the RGB arm.
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8983
export MAIN_PORT=8983
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
eval "$(grep 'WANDB_API_KEY' ~/.bashrc)"
# self-contained env activation (tmux/cron shells start in base)
source ~/miniconda/etc/profile.d/conda.sh && conda activate pirlnav
cd ~/pirlnav

python -u -m run \
    --exp-config configs/experiments/il_objectnav_costmap.yaml \
    --run-type train \
    TENSORBOARD_DIR tb/il_costmap_2ch \
    CHECKPOINT_FOLDER data/new_checkpoints/il_costmap_2ch \
    NUM_UPDATES 25000 \
    NUM_ENVIRONMENTS 10 \
    NUM_ACCUM_STEPS 51 \
    IL.BehaviorCloning.num_steps 32 \
    IL.BehaviorCloning.num_mini_batch 5 \
    NUM_CHECKPOINTS 10 \
    LOG_INTERVAL 50 \
    WRITER_TYPE wb \
    WB.PROJECT_NAME pirlnav-baseline \
    WB.RUN_NAME il_costmap_2ch_10env_acc51 \
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS 10000 \
    TASK_CONFIG.DATASET.DATA_PATH "data/datasets/objectnav/objectnav_hm3d_hd/{split}/{split}.json.gz" \
    TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.234951275740812
