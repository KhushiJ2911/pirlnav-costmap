#!/bin/bash
# Eval the overfit checkpoint on the SAME scene/episodes it memorized (split=train).
# High SR  -> eval harness certified (loading, act(), STOP, success measure all work).
# SR ~ 0 despite train acc ~1 -> real eval-side bug: dig into STOP/action mapping.
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8973
export CUDA_VISIBLE_DEVICES=${EVAL_GPU:-1}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
# WANDB_API_KEY lives in ~/.bashrc, which non-interactive shells skip -> pull it in
eval "$(grep 'WANDB_API_KEY' ~/.bashrc)"
cd ~/pirlnav

CKPT=${1:-data/new_checkpoints/overfit_long/ckpt.2.pth}

python -u -m run \
    --exp-config configs/experiments/il_objectnav.yaml \
    --run-type eval \
    TENSORBOARD_DIR tb/overfit_long_eval \
    WRITER_TYPE wb \
    WB.PROJECT_NAME pirlnav-baseline \
    WB.RUN_NAME overfit_eval_same_scene \
    EVAL_CKPT_PATH_DIR "$CKPT" \
    NUM_ENVIRONMENTS 2 \
    TEST_EPISODE_COUNT -1 \
    EVAL.SPLIT train \
    EVAL.DATASET_TYPE ObjectNav-v2 \
    EVAL.USE_CKPT_CONFIG False \
    POLICY.RGB_ENCODER.pretrained_encoder data/visual_encoders/omnidata_DINO_02.pth \
    POLICY.RGB_ENCODER.use_augmentations False \
    POLICY.RGB_ENCODER.use_augmentations_test_time False \
    TASK_CONFIG.DATASET.SPLIT train \
    TASK_CONFIG.DATASET.DATA_PATH "data/datasets/objectnav/overfit_tiny/{split}/{split}.json.gz"
