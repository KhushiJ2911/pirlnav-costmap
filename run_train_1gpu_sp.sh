#!/bin/bash
# Experiment 2, RGB arm: identical certified recipe to run_train_1gpu.sh with the
# teacher swapped: shortest-path demos (objectnav_hm3d_sp_gen) instead of human
# demos. Paper's prediction: SP demos are a BAD teacher for RGB (6.4 SR at 240k
# in the PIRLNav paper) — this arm is the control in the interaction-effect test.
# Set INFLECTION_COEF from run_gen_sp_demos.sh's final report before launching.
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=8991
export MAIN_PORT=8991
export CUDA_VISIBLE_DEVICES=${TRAIN_GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
eval "$(grep 'WANDB_API_KEY' ~/.bashrc)"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda/etc/profile.d/conda.sh
conda activate pirlnav
cd ~/pirlnav

: "${INFLECTION_COEF:?Set INFLECTION_COEF (from run_gen_sp_demos.sh --report-coef output)}"

python -u -m run \
    --exp-config configs/experiments/il_objectnav.yaml \
    --run-type train \
    TENSORBOARD_DIR tb/il_rgb_sp \
    CHECKPOINT_FOLDER data/new_checkpoints/il_rgb_sp \
    NUM_UPDATES 25000 \
    NUM_ENVIRONMENTS 10 \
    NUM_ACCUM_STEPS 51 \
    IL.BehaviorCloning.num_steps 32 \
    IL.BehaviorCloning.num_mini_batch 5 \
    NUM_CHECKPOINTS 10 \
    LOG_INTERVAL 50 \
    WRITER_TYPE wb \
    WB.PROJECT_NAME pirlnav-baseline \
    WB.RUN_NAME il_rgb_sp_10env_acc51 \
    POLICY.RGB_ENCODER.pretrained_encoder data/visual_encoders/omnidata_DINO_02.pth \
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS 10000 \
    TASK_CONFIG.DATASET.DATA_PATH "data/datasets/objectnav/objectnav_hm3d_sp_gen/{split}/{split}.json.gz" \
    TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF "$INFLECTION_COEF"
