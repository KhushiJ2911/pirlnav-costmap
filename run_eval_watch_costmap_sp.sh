#!/bin/bash
# Experiment 2 LIVE eval, costmap-SP arm (the headline measurement): start this
# right after training starts. Polls the checkpoint FOLDER and evaluates every
# checkpoint as it appears on the FULL 1000-ep val split, logging to W&B against
# training step — the SR-vs-frames curve builds while training runs. NOTE: the
# poll loop never exits; kill the tmux session after ckpt.9 has been evaluated.
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export CUDA_VISIBLE_DEVICES=${EVAL_GPU:-3}
export MASTER_PORT=8996
export MAIN_PORT=8996
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
eval "$(grep 'WANDB_API_KEY' ~/.bashrc)"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda/etc/profile.d/conda.sh
conda activate pirlnav
cd ~/pirlnav

python -u -m run \
  --exp-config configs/experiments/il_objectnav_costmap.yaml \
  --run-type eval \
  TENSORBOARD_DIR tb/il_costmap_2ch_sp_eval \
  EVAL_CKPT_PATH_DIR data/new_checkpoints/il_costmap_2ch_sp \
  NUM_ENVIRONMENTS ${EVAL_ENVS:-6} \
  TEST_EPISODE_COUNT -1 \
  EVAL.USE_CKPT_CONFIG False \
  WRITER_TYPE wb \
  WB.PROJECT_NAME pirlnav-baseline \
  WB.RUN_NAME il_costmap_2ch_sp_eval \
  TASK_CONFIG.DATASET.SPLIT val \
  TASK_CONFIG.DATASET.DATA_PATH "data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2/{split}/{split}.json.gz"
