#!/bin/bash
# Experiment 2 LIVE eval, RGB-SP arm: start this right after training starts.
# Points habitat eval at the checkpoint FOLDER: it polls and evaluates every
# checkpoint (existing + new) as it appears, on the FULL 1000-ep val split,
# logging eval_metrics/* to W&B against each checkpoint's training step. First
# behavioral signal lands ~3-5h after training start (ckpt.0 + eval time) — no
# need to wait for training to finish. NOTE: the poll loop never exits; kill
# the tmux session once the last checkpoint (ckpt.9) has been evaluated.
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export CUDA_VISIBLE_DEVICES=${EVAL_GPU:-2}
export MASTER_PORT=8995
export MAIN_PORT=8995
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
eval "$(grep 'WANDB_API_KEY' ~/.bashrc)"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda/etc/profile.d/conda.sh
conda activate pirlnav
cd ~/pirlnav

python -u -m run \
  --exp-config configs/experiments/il_objectnav.yaml \
  --run-type eval \
  TENSORBOARD_DIR tb/il_rgb_sp_eval \
  EVAL_CKPT_PATH_DIR data/new_checkpoints/il_rgb_sp \
  NUM_ENVIRONMENTS ${EVAL_ENVS:-6} \
  TEST_EPISODE_COUNT -1 \
  EVAL.USE_CKPT_CONFIG False \
  RL.DDPPO.force_distributed True \
  WRITER_TYPE wb \
  WB.PROJECT_NAME pirlnav-baseline \
  WB.RUN_NAME il_rgb_sp_eval \
  TASK_CONFIG.DATASET.SPLIT val \
  TASK_CONFIG.DATASET.DATA_PATH "data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2/{split}/{split}.json.gz"
