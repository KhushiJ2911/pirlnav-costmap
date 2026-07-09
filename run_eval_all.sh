#!/bin/bash

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8954
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TMPDIR=$HOME/tmp

mkdir -p "$TMPDIR"
cd ~/pirlnav

python -u -m run \
  --exp-config configs/experiments/il_objectnav.yaml \
  --run-type eval \
  TENSORBOARD_DIR tb/il_repro_eval \
  EVAL_CKPT_PATH_DIR data/new_checkpoints/il_repro \
  NUM_ENVIRONMENTS 4 \
  TEST_EPISODE_COUNT -1 \
  WRITER_TYPE wb \
  WB.PROJECT_NAME pirlnav-baseline \
  WB.RUN_NAME il_ovrl_repro20k_eval \
  RL.DDPPO.force_distributed True \
  TASK_CONFIG.DATASET.SPLIT val \
  EVAL.USE_CKPT_CONFIG False \
  POLICY.RGB_ENCODER.pretrained_encoder data/visual_encoders/omnidata_DINO_02.pth \
  TASK_CONFIG.DATASET.DATA_PATH "data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2/{split}/{split}.json.gz"