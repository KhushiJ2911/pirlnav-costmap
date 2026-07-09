#!/bin/bash
# Overfit 1 scene until memorized (accum=1, frequent steps), then eval same scene
# with run_eval_overfit.sh. Purpose: certify the EVAL harness (handoff sec 12).
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8972
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
# WANDB_API_KEY lives in ~/.bashrc, which non-interactive shells skip -> pull it in
eval "$(grep 'WANDB_API_KEY' ~/.bashrc)"
cd ~/pirlnav

python -u -m run \
    --exp-config configs/experiments/il_objectnav.yaml \
    --run-type train \
    TENSORBOARD_DIR tb/overfit_long \
    CHECKPOINT_FOLDER data/new_checkpoints/overfit_long \
    NUM_UPDATES 2000 \
    NUM_ENVIRONMENTS 2 \
    NUM_ACCUM_STEPS 1 \
    IL.BehaviorCloning.num_mini_batch 1 \
    NUM_CHECKPOINTS 4 \
    LOG_INTERVAL 20 \
    WRITER_TYPE wb \
    WB.PROJECT_NAME pirlnav-baseline \
    WB.RUN_NAME il_overfit_long_2000 \
    POLICY.RGB_ENCODER.pretrained_encoder data/visual_encoders/omnidata_DINO_02.pth \
    POLICY.RGB_ENCODER.use_augmentations False \
    POLICY.RGB_ENCODER.use_augmentations_test_time False \
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.DATASET.DATA_PATH "data/datasets/objectnav/overfit_tiny/{split}/{split}.json.gz" \
    TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.234951275740812
