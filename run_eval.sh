#!/bin/bash

export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8951
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"

cd ~/pirlnav

python -u -m run \
  --exp-config configs/experiments/il_objectnav.yaml \
  --run-type eval \
  TENSORBOARD_DIR tb/il_long_eval \
  EVAL_CKPT_PATH_DIR data/new_checkpoints/il_long/ckpt.4.pth \
  NUM_ENVIRONMENTS 4 \
  TEST_EPISODE_COUNT -1 \
  RL.DDPPO.force_distributed True \
    WRITER_TYPE wb \
  WB.PROJECT_NAME pirlnav-repro \
  WB.RUN_NAME il_long_eval \
  TASK_CONFIG.DATASET.SPLIT val \
  EVAL.USE_CKPT_CONFIG False \
  TASK_CONFIG.DATASET.DATA_PATH "data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2/{split}/{split}.json.gz"