#!/bin/bash
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8960
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
cd ~/pirlnav

python -u -m run \
    --exp-config configs/experiments/il_objectnav.yaml \
    --run-type train \
    TENSORBOARD_DIR tb/il_gradaccum \
    CHECKPOINT_FOLDER data/new_checkpoints/il_gradaccum \
    NUM_UPDATES 3200 \
    NUM_ENVIRONMENTS 8 \
    NUM_ACCUM_STEPS 32 \
    NUM_CHECKPOINTS 4 \
    LOG_INTERVAL 10 \
    WRITER_TYPE wb \
    WB.PROJECT_NAME pirlnav-baseline \
    WB.RUN_NAME il_gradaccum_8env_acc32 \
    POLICY.RGB_ENCODER.pretrained_encoder data/visual_encoders/omnidata_DINO_02.pth \
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.DATASET.DATA_PATH "data/datasets/objectnav/objectnav_hm3d_hd/{split}/{split}.json.gz" \
    TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 3.234951275740812
