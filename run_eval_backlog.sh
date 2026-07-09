#!/bin/bash
# RGB baseline eval backlog: evaluates ckpt.3..9 of the 1-day RGB run, one file
# at a time (pointing at the folder would restart from ckpt.0). Sequential
# invocations resume the SAME W&B run (wrj46kdj) so all points land on the
# existing eval curve. Full 1000-episode val each (~2.5-4h per ckpt).
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export CUDA_VISIBLE_DEVICES=${EVAL_GPU:-1}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
eval "$(grep 'WANDB_API_KEY' ~/.bashrc)"
source ~/miniconda/etc/profile.d/conda.sh && conda activate pirlnav
cd ~/pirlnav
export WANDB_RUN_ID=wrj46kdj WANDB_RESUME=allow

for i in 3 4 5 6 7 8 9; do
  CKPT=data/new_checkpoints/il_1gpu_10env/ckpt.$i.pth
  echo "=== $(date '+%m-%d %H:%M') evaluating $CKPT ==="
  python -u -m run \
    --exp-config configs/experiments/il_objectnav.yaml \
    --run-type eval \
    TENSORBOARD_DIR tb/il_rgb_1day_eval \
    EVAL_CKPT_PATH_DIR "$CKPT" \
    NUM_ENVIRONMENTS 4 \
    TEST_EPISODE_COUNT -1 \
    EVAL.USE_CKPT_CONFIG False \
    WRITER_TYPE wb \
    WB.PROJECT_NAME pirlnav-baseline \
    WB.RUN_NAME il_rgb_1day_eval \
    TASK_CONFIG.DATASET.SPLIT val \
    TASK_CONFIG.DATASET.DATA_PATH "data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2/{split}/{split}.json.gz"
done
echo "EVAL-BACKLOG-COMPLETE"
