#!/bin/bash
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8963
export CUDA_VISIBLE_DEVICES=2    # a DIFFERENT free GPU from training
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
cd ~/pirlnav

python -u -m run \
    --exp-config configs/experiments/il_objectnav.yaml \
    --run-type eval \
    TENSORBOARD_DIR tb/il_overfit_eval \
    EVAL_CKPT_PATH_DIR data/new_checkpoints/il_overfit \
    NUM_ENVIRONMENTS 2 \
    TEST_EPISODE_COUNT 30 \
    WRITER_TYPE wb \
    WB.PROJECT_NAME pirlnav-baseline \
    WB.RUN_NAME il_overfit_1scene_eval \
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.DATASET.SPLIT train \
    TASK_CONFIG.DATASET.CONTENT_SCENES "['1S7LAXRdDqK']" \
    EVAL.USE_CKPT_CONFIG False \
    POLICY.RGB_ENCODER.pretrained_encoder data/visual_encoders/omnidata_DINO_02.pth \
    TASK_CONFIG.DATASET.DATA_PATH "data/datasets/objectnav/objectnav_hm3d_hd/{split}/{split}.json.gz"