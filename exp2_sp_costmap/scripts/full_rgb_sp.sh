#!/bin/bash
export LC_ALL=C GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export MASTER_ADDR=127.0.0.1 MASTER_PORT=8978 MAIN_PORT=8978
export CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TMPDIR=\$HOME/tmp; mkdir -p "\$TMPDIR"
eval "\$(grep WANDB_API_KEY ~/.bashrc)"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pirlnav
cd ~/pirlnav
python -u -m run --exp-config configs/experiments/il_objectnav.yaml --run-type train \
  TENSORBOARD_DIR tb/il_rgb_sp CHECKPOINT_FOLDER data/new_checkpoints/il_rgb_sp \
  NUM_UPDATES 25000 NUM_ENVIRONMENTS 10 NUM_ACCUM_STEPS 51 \
  IL.BehaviorCloning.num_steps 32 IL.BehaviorCloning.num_mini_batch 5 \
  NUM_CHECKPOINTS 10 LOG_INTERVAL 50 WRITER_TYPE wb \
  WB.PROJECT_NAME pirlnav-baseline WB.RUN_NAME il_rgb_sp \
  POLICY.RGB_ENCODER.pretrained_encoder data/visual_encoders/omnidata_DINO_02.pth \
  RL.DDPPO.force_distributed True \
  TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS 10000 \
  TASK_CONFIG.DATASET.DATA_PATH "data/datasets/objectnav/objectnav_hm3d_sp_gen/{split}/{split}.json.gz" \
  TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF 1.695062344794965
echo "FULL-RGB-SP-EXIT-\$?"
