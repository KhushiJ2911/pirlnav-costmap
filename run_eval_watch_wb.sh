#!/bin/bash
# Eval watcher: waits for a FREE GPU (0 or 2; checks every 2 min, claims only when
# another user's job has ended: <500MiB used). Then points habitat eval at the
# checkpoint FOLDER of the 1-day RGB run: habitat polls the folder and evaluates
# every checkpoint (existing + new) on the FULL 1000-episode val split, logging
# eval_metrics/* to W&B (run il_rgb_1day_eval) against each checkpoint's training
# step -> val curves alongside the training run. If no GPU frees overnight, it
# simply starts later; the resulting W&B curve is identical (x-axis is ckpt step,
# not wall time).
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
eval "$(grep 'WANDB_API_KEY' ~/.bashrc)"
# self-contained env activation (tmux/cron shells start in base)
source ~/miniconda/etc/profile.d/conda.sh && conda activate pirlnav
cd ~/pirlnav

GPU=""
while [ -z "$GPU" ]; do
  for g in 0 2; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    if [ "$USED" -lt 500 ]; then GPU=$g; break; fi
  done
  [ -z "$GPU" ] && { echo "$(date '+%H:%M') GPUs 0/2 busy, retrying in 120s"; sleep 120; }
done
echo "$(date '+%H:%M') claiming GPU $GPU for eval watcher"
export CUDA_VISIBLE_DEVICES=$GPU

python -u -m run \
  --exp-config configs/experiments/il_objectnav.yaml \
  --run-type eval \
  TENSORBOARD_DIR tb/il_rgb_1day_eval \
  EVAL_CKPT_PATH_DIR data/new_checkpoints/il_1gpu_10env \
  NUM_ENVIRONMENTS 4 \
  TEST_EPISODE_COUNT -1 \
  EVAL.USE_CKPT_CONFIG False \
  WRITER_TYPE wb \
  WB.PROJECT_NAME pirlnav-baseline \
  WB.RUN_NAME il_rgb_1day_eval \
  TASK_CONFIG.DATASET.SPLIT val \
  TASK_CONFIG.DATASET.DATA_PATH "data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2/{split}/{split}.json.gz"
