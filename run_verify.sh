#!/bin/bash
# Pipeline verification: overfit a single scene with gradient accumulation.
# If the training loop + grad-accum + metrics are wired correctly, action_ce_loss
# should fall and action_accuracy should climb over the run.
# 80 updates / NUM_ACCUM_STEPS 4 = 20 real optimizer steps.
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8971
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
cd ~/pirlnav

python -u -m run \
    --exp-config configs/experiments/il_objectnav.yaml \
    --run-type train \
    TENSORBOARD_DIR tb/verify \
    CHECKPOINT_FOLDER data/new_checkpoints/verify \
    NUM_UPDATES 80 \
    NUM_ENVIRONMENTS 2 \
    NUM_ACCUM_STEPS 4 \
    IL.BehaviorCloning.num_mini_batch 1 \
    NUM_CHECKPOINTS 1 \
    LOG_INTERVAL 1 \
    WRITER_TYPE tb \
    POLICY.RGB_ENCODER.pretrained_encoder data/visual_encoders/omnidata_DINO_02.pth \
    POLICY.RGB_ENCODER.use_augmentations False \
    POLICY.RGB_ENCODER.use_augmentations_test_time False \
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.DATASET.DATA_PATH "data/datasets/objectnav/overfit_tiny/{split}/{split}.json.gz" \
    TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.234951275740812
