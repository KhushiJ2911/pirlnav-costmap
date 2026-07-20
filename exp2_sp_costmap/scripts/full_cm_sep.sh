#!/bin/bash
export LC_ALL=C GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export MASTER_ADDR=127.0.0.1 MASTER_PORT=8977 MAIN_PORT=8977
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SIM_GPU_ID=3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TMPDIR=$HOME/tmp; mkdir -p "$TMPDIR"
eval "$(grep WANDB_API_KEY ~/.bashrc)"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pirlnav
cd ~/pirlnav
python -u -m run --exp-config configs/experiments/il_objectnav_costmap.yaml --run-type train \
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
  TASK_CONFIG.DATASET.DATA_PATH "data/datasets/objectnav/objectnav_hm3d_sp_gen/{split}/{split}.json.gz" \
  TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 1.695062344794965
echo "STABLE-CM-EXIT-$?"
