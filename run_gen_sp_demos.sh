#!/bin/bash
# Experiment 2 step 1: generate shortest-path demos for all 80 HD train scenes.
# CPU-bound; shards across processes. Safe to re-run: finished scenes are skipped.
# Usage: bash run_gen_sp_demos.sh [NUM_SHARDS]
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR" logs
eval "$(grep 'WANDB_API_KEY' ~/.bashrc)"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda/etc/profile.d/conda.sh
conda activate pirlnav
cd ~/pirlnav

NUM_SHARDS=${1:-8}
pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  python -u generate_sp_demos.py \
      --num-shards "$NUM_SHARDS" --shard-index "$i" \
      > "logs/gen_sp_s${i}.log" 2>&1 &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail -ne 0 ] && echo "WARNING: at least one shard failed; check logs/gen_sp_s*.log"

echo "=== global inflection coefficient (pass it to the SP training scripts) ==="
python generate_sp_demos.py --report-coef data/datasets/objectnav/objectnav_hm3d_sp_gen
