#!/bin/bash
# Self-healing costmap training: auto-resumes from the last checkpoint after any
# CUDA crash, and kills+resumes on any hang (CPU-spin), marching to NUM_UPDATES
# across as many restarts as needed. Resume is via habitat's .habitat-resume-state.
export LC_ALL=C GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export MASTER_ADDR=127.0.0.1 MASTER_PORT=8977 MAIN_PORT=8977
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SIM_GPU_ID=3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TMPDIR=$HOME/tmp; mkdir -p "$TMPDIR"
eval "$(grep WANDB_API_KEY ~/.bashrc)"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pirlnav
cd ~/pirlnav

LOG=~/full_cm_sp.log
TARGET=24900
HANG_CYCLES=6          # 6 x 60s = ~6 min of no update-progress => treat as hang
FASTFAIL_LIMIT=6       # consecutive <90s attempts with no progress => give up

PYCMD="python -u -m run --exp-config configs/experiments/il_objectnav_costmap.yaml --run-type train \
  IL.BehaviorCloning.lr 0.0005 IL.BehaviorCloning.encoder_lr 0.0005 IL.BehaviorCloning.max_grad_norm 0.1 \
  TENSORBOARD_DIR tb/il_costmap_2ch_sp CHECKPOINT_FOLDER data/new_checkpoints/il_costmap_2ch_sp \
  NUM_UPDATES 25000 NUM_ENVIRONMENTS 10 NUM_ACCUM_STEPS 51 \
  IL.BehaviorCloning.num_steps 32 IL.BehaviorCloning.num_mini_batch 5 \
  NUM_CHECKPOINTS 10 LOG_INTERVAL 50 WRITER_TYPE wb \
  WB.PROJECT_NAME pirlnav-baseline WB.RUN_NAME il_costmap_2ch_sp \
  RL.DDPPO.force_distributed True \
  POLICY.COSTMAP_ENCODER.normalize_visual_inputs False \
  TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS 10000 \
  TASK_CONFIG.TASK.COSTMAP_SENSOR.OUT_HEIGHT 32 TASK_CONFIG.TASK.COSTMAP_SENSOR.OUT_WIDTH 32 \
  TASK_CONFIG.DATASET.DATA_PATH data/datasets/objectnav/objectnav_hm3d_sp_gen/{split}/{split}.json.gz \
  TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 1.695062344794965"

cur_update() { grep -oE "update: [0-9]+" $LOG | tail -1 | grep -oE "[0-9]+"; }

attempt=0
fastfail=0
while true; do
  attempt=$((attempt+1))
  start_u=$(cur_update); start_u=${start_u:-0}
  t0=$(date +%s)
  echo "===== RESILIENT ATTEMPT $attempt @ $(date '+%H:%M:%S') from update ~$start_u =====" >> $LOG
  $PYCMD >> $LOG 2>&1 &
  PYPID=$!

  last=""; stuck=0
  while kill -0 $PYPID 2>/dev/null; do
    sleep 60
    cur=$(cur_update)
    if [ -n "$cur" ] && [ "$cur" = "$last" ]; then stuck=$((stuck+1)); else stuck=0; fi
    if [ "$stuck" -ge "$HANG_CYCLES" ]; then
      echo "===== HANG at update $cur (no progress ~${HANG_CYCLES}min) -> killing to resume =====" >> $LOG
      kill -9 $PYPID 2>/dev/null
      break
    fi
    last=$cur
  done
  wait $PYPID 2>/dev/null
  pkill -9 -f il_objectnav_costmap.yaml 2>/dev/null
  sleep 4

  end_u=$(cur_update); end_u=${end_u:-0}
  dt=$(( $(date +%s) - t0 ))
  echo "===== attempt $attempt ended @ $(date '+%H:%M:%S'): update $start_u -> $end_u in ${dt}s =====" >> $LOG

  if [ "$end_u" -ge "$TARGET" ]; then
    echo "===== COSTMAP COMPLETE at update $end_u =====" >> $LOG
    break
  fi
  # fast-fail guard: crash-loop with no progress => stop
  if [ "$dt" -lt 90 ] && [ "$end_u" -le "$start_u" ]; then
    fastfail=$((fastfail+1))
  else
    fastfail=0
  fi
  if [ "$fastfail" -ge "$FASTFAIL_LIMIT" ]; then
    echo "===== ABORT: $FASTFAIL_LIMIT consecutive fast-fails with no progress (GPU/config broken) =====" >> $LOG
    break
  fi
  sleep 5
done
echo "RESILIENT-CM-EXIT" >> $LOG
