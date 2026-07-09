#!/bin/bash
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8964
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
cd ~/pirlnav

python -u -m run \
    --exp-config configs/experiments/il_objectnav.yaml \
    --run-type train \
    TENSORBOARD_DIR tb/il_overfit_tiny \
    CHECKPOINT_FOLDER data/new_checkpoints/il_overfit_tiny \
    NUM_UPDATES 3000 \
    NUM_ENVIRONMENTS 1 \
    NUM_ACCUM_STEPS 1 \
    IL.BehaviorCloning.num_mini_batch 1 \
    NUM_CHECKPOINTS 5 \
    LOG_INTERVAL 5 \
    WRITER_TYPE wb \
    WB.PROJECT_NAME pirlnav-baseline \
    WB.RUN_NAME il_overfit_5ep \
    POLICY.RGB_ENCODER.pretrained_encoder data/visual_encoders/omnidata_DINO_02.pth \
    POLICY.RGB_ENCODER.use_augmentations False \
    POLICY.RGB_ENCODER.use_augmentations_test_time False \
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.DATASET.DATA_PATH "data/datasets/objectnav/overfit_tiny/{split}/{split}.json.gz" \
    TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.234951275740812