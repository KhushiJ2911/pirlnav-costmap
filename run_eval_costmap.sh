#!/bin/bash
# Costmap-arm eval: ckpts {1,3,5,7,9} of the 2-channel run on the FULL
# 1000-episode HM3D-v2 val split (sequential, one file at a time; ~2.5-4h each).
# All points land on one W&B run (il_costmap_2ch_eval) -> SR/softSPL-vs-frames
# curve directly overlayable with the RGB eval curve (wrj46kdj).
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export CUDA_VISIBLE_DEVICES=${EVAL_GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
eval "$(grep 'WANDB_API_KEY' ~/.bashrc)"
source ~/miniconda/etc/profile.d/conda.sh && conda activate pirlnav
cd ~/pirlnav
export WANDB_RUN_ID=cm2cheval1 WANDB_RESUME=allow

for i in 1 3 5 7 9; do
  CKPT=data/new_checkpoints/il_costmap_2ch/ckpt.$i.pth
  echo "=== $(date '+%m-%d %H:%M') evaluating $CKPT ==="
  python -u -m run \
    --exp-config configs/experiments/il_objectnav_costmap.yaml \
    --run-type eval \
    TENSORBOARD_DIR tb/il_costmap_2ch_eval \
    EVAL_CKPT_PATH_DIR "$CKPT" \
    NUM_ENVIRONMENTS ${EVAL_ENVS:-10} \
    TEST_EPISODE_COUNT -1 \
    EVAL.USE_CKPT_CONFIG False \
    WRITER_TYPE wb \
    WB.PROJECT_NAME pirlnav-baseline \
    WB.RUN_NAME il_costmap_2ch_eval \
    TASK_CONFIG.DATASET.SPLIT val \
    TASK_CONFIG.DATASET.DATA_PATH "data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2/{split}/{split}.json.gz"
done
echo "COSTMAP-EVAL-COMPLETE"
