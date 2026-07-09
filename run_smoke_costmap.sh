#!/bin/bash
# Costmap-arm smoke: 10 updates on 1 scene. Verifies CostmapSensor -> rollout ->
# ObjectNavILCostmapPolicy end-to-end, and logs geodesic field build time (cached
# per (scene, category) thereafter).
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8982
export MAIN_PORT=8982  # habitat ddp_utils reads MAIN_PORT; default 8738 collides with concurrent runs
export CUDA_VISIBLE_DEVICES=${SMOKE_GPU:-2}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
cd ~/pirlnav

python -u -m run \
    --exp-config configs/experiments/il_objectnav_costmap.yaml \
    --run-type train \
    TENSORBOARD_DIR tb/smoke_costmap \
    CHECKPOINT_FOLDER data/new_checkpoints/smoke_costmap \
    NUM_UPDATES 10 \
    NUM_ENVIRONMENTS 2 \
    NUM_ACCUM_STEPS 1 \
    IL.BehaviorCloning.num_mini_batch 1 \
    NUM_CHECKPOINTS 1 \
    LOG_INTERVAL 1 \
    WRITER_TYPE tb \
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.DATASET.DATA_PATH "data/datasets/objectnav/overfit_tiny/{split}/{split}.json.gz" \
    TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.234951275740812
