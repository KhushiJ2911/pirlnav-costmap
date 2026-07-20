#!/bin/bash
# Eval RGB-SP arm (il_rgb_sp) on the FULL 1000-ep HM3D-v2 val split, GPU 2.
# Checkpoints {1,3,5,7,9} = updates {5k,10k,15k,20k,25k}. Logs to W&B.
# EVAL.DATASET_TYPE ObjectNav-v2 is required for this v2-format val split.
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export CUDA_VISIBLE_DEVICES=2
export MASTER_ADDR=127.0.0.1 MASTER_PORT=8993 MAIN_PORT=8993
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TMPDIR=$HOME/tmp; mkdir -p "$TMPDIR"
eval "$(grep 'WANDB_API_KEY' ~/.bashrc)"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pirlnav
cd ~/pirlnav

for i in 9 7 5 3 1; do
  CKPT=data/new_checkpoints/il_rgb_sp/ckpt.$i.pth
  [ -f "$CKPT" ] || { echo "skip missing $CKPT"; continue; }
  echo "===== $(date '+%m-%d %H:%M') EVAL il_rgb_sp ckpt.$i ====="
  python -u -m run \
    --exp-config configs/experiments/il_objectnav.yaml \
    --run-type eval \
    TENSORBOARD_DIR tb/il_rgb_sp_eval \
    EVAL_CKPT_PATH_DIR "$CKPT" \
    NUM_ENVIRONMENTS 10 \
    TEST_EPISODE_COUNT -1 \
    EVAL.USE_CKPT_CONFIG False \
    EVAL.DATASET_TYPE ObjectNav-v2 \
    RL.DDPPO.force_distributed True \
    WRITER_TYPE wb \
    WB.PROJECT_NAME pirlnav-baseline \
    WB.RUN_NAME il_rgb_sp_eval_ckpt$i \
    TASK_CONFIG.DATASET.SPLIT val \
    TASK_CONFIG.DATASET.DATA_PATH "data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2/{split}/{split}.json.gz"
  echo "===== ckpt.$i done @ $(date '+%H:%M') ====="
done
echo "RGB-SP-EVAL-COMPLETE"
