#!/bin/bash
# Costmap-SP eval on the FULL 1000-ep HM3D-v2 val, GPU 0. THE headline number.
# Config MUST match training: 32x32 costmap, normalize off, ObjectNav-v2 val.
# Sensor loads precomputed VAL fields from data/costmap_field_cache (must be
# precomputed first, else Euclidean fallback everywhere). ckpt.9 first.
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=127.0.0.1 MASTER_PORT=8996 MAIN_PORT=8996
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TMPDIR=$HOME/tmp; mkdir -p "$TMPDIR"
eval "$(grep 'WANDB_API_KEY' ~/.bashrc)"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pirlnav
cd ~/pirlnav

for i in 9 7 5 3 1; do
  CKPT=data/new_checkpoints/il_costmap_2ch_sp/ckpt.$i.pth
  [ -f "$CKPT" ] || { echo "skip missing $CKPT"; continue; }
  echo "===== $(date '+%m-%d %H:%M') EVAL il_costmap_2ch_sp ckpt.$i ====="
  python -u -m run \
    --exp-config configs/experiments/il_objectnav_costmap.yaml \
    --run-type eval \
    TENSORBOARD_DIR tb/il_costmap_2ch_sp_eval \
    EVAL_CKPT_PATH_DIR "$CKPT" \
    NUM_ENVIRONMENTS 10 \
    TEST_EPISODE_COUNT -1 \
    EVAL.USE_CKPT_CONFIG False \
    EVAL.DATASET_TYPE ObjectNav-v2 \
    RL.DDPPO.force_distributed True \
    POLICY.COSTMAP_ENCODER.normalize_visual_inputs False \
    TASK_CONFIG.TASK.COSTMAP_SENSOR.OUT_HEIGHT 32 \
    TASK_CONFIG.TASK.COSTMAP_SENSOR.OUT_WIDTH 32 \
    WRITER_TYPE wb \
    WB.PROJECT_NAME pirlnav-baseline \
    WB.RUN_NAME il_costmap_2ch_sp_eval_ckpt$i \
    TASK_CONFIG.DATASET.SPLIT val \
    TASK_CONFIG.DATASET.DATA_PATH "data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2/{split}/{split}.json.gz"
  echo "===== ckpt.$i done @ $(date '+%H:%M') ====="
done
echo "COSTMAP-SP-EVAL-COMPLETE"
