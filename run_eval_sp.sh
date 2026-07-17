#!/bin/bash
# Experiment 2 eval, RGB-SP arm: ckpts {1,3,5,7,9} of il_rgb_sp on the FULL
# 1000-episode HM3D-v2 val split. Same protocol as run_eval_costmap.sh.
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export CUDA_VISIBLE_DEVICES=${EVAL_GPU:-2}
export MASTER_PORT=8993
export MAIN_PORT=8993
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
eval "$(grep 'WANDB_API_KEY' ~/.bashrc)"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda/etc/profile.d/conda.sh
conda activate pirlnav
cd ~/pirlnav

for i in 1 3 5 7 9; do
  CKPT=data/new_checkpoints/il_rgb_sp/ckpt.$i.pth
  [ -f "$CKPT" ] || { echo "skip missing $CKPT"; continue; }
  echo "=== $(date '+%m-%d %H:%M') evaluating $CKPT ==="
  python -u -m run \
    --exp-config configs/experiments/il_objectnav.yaml \
    --run-type eval \
    TENSORBOARD_DIR tb/il_rgb_sp_eval \
    EVAL_CKPT_PATH_DIR "$CKPT" \
    NUM_ENVIRONMENTS ${EVAL_ENVS:-10} \
    TEST_EPISODE_COUNT -1 \
    EVAL.USE_CKPT_CONFIG False \
    RL.DDPPO.force_distributed True \
    WRITER_TYPE wb \
    WB.PROJECT_NAME pirlnav-baseline \
    WB.RUN_NAME il_rgb_sp_eval \
    TASK_CONFIG.DATASET.SPLIT val \
    TASK_CONFIG.DATASET.DATA_PATH "data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2/{split}/{split}.json.gz"
done
echo "RGB-SP-EVAL-COMPLETE"
