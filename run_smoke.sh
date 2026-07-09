#!/bin/bash

export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8949
export CUDA_VISIBLE_DEVICES=0
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"

cd ~/pirlnav

python -u -m run \
  --exp-config configs/experiments/il_objectnav.yaml \
  --run-type train \
  TENSORBOARD_DIR tb/smoke \
  CHECKPOINT_FOLDER data/new_checkpoints/smoke \
  NUM_UPDATES 20 \
  NUM_ENVIRONMENTS 2 \
  NUM_CHECKPOINTS 1 \
  LOG_INTERVAL 1 \
  RL.DDPPO.force_distributed True \
  TASK_CONFIG.DATASET.DATA_PATH "data/datasets/objectnav/objectnav_hm3d_hd/{split}/{split}.json.gz" \
  TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.234951275740812