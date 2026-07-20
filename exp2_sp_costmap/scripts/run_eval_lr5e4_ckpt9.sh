#!/bin/bash
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export CUDA_VISIBLE_DEVICES=1
export MASTER_ADDR=127.0.0.1 MASTER_PORT=8995 MAIN_PORT=8995
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TMPDIR=$HOME/tmp; mkdir -p "$TMPDIR"
eval "$(grep 'WANDB_API_KEY' ~/.bashrc)"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pirlnav
cd ~/pirlnav
CKPT=data/new_checkpoints/il_rgb_sp_lr5e4/ckpt.9.pth
echo "===== $(date '+%H:%M') EVAL il_rgb_sp_lr5e4 ckpt.9 (GPU1) ====="
python -u -m run --exp-config configs/experiments/il_objectnav.yaml --run-type eval \
  TENSORBOARD_DIR tb/il_rgb_lr5e4_eval EVAL_CKPT_PATH_DIR "$CKPT" \
  NUM_ENVIRONMENTS 12 TEST_EPISODE_COUNT -1 EVAL.USE_CKPT_CONFIG False \
  EVAL.DATASET_TYPE ObjectNav-v2 RL.DDPPO.force_distributed True \
  WRITER_TYPE wb WB.PROJECT_NAME pirlnav-baseline WB.RUN_NAME il_rgb_lr5e4_eval_ckpt9 \
  TASK_CONFIG.DATASET.SPLIT val \
  TASK_CONFIG.DATASET.DATA_PATH "data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2/{split}/{split}.json.gz"
echo "===== lr5e4 ckpt.9 done @ $(date '+%H:%M') ====="
