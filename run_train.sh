#!/bin/bash

export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8950
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"

cd ~/pirlnav

python -u -m run \
  --exp-config configs/experiments/il_objectnav.yaml \
  --run-type train \
  TENSORBOARD_DIR tb/il_long \
  CHECKPOINT_FOLDER data/new_checkpoints/il_long \
  NUM_UPDATES 2000 \
  NUM_ENVIRONMENTS 4 \
  NUM_CHECKPOINTS 5 \
  LOG_INTERVAL 10 \
  WRITER_TYPE wb \
  WB.PROJECT_NAME pirlnav-repro \
  WB.RUN_NAME il_long_run \
  RL.DDPPO.force_distributed True \
  TASK_CONFIG.DATASET.DATA_PATH "data/datasets/objectnav/objectnav_hm3d_hd/{split}/{split}.json.gz" \
  TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.234951275740812