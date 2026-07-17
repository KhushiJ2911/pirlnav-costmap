#!/bin/bash
# Experiment 1: greedy-descent probe on full HM3D-v2 val (eval-only, no training).
# Shards across 4 processes by scene; merge + summary printed at the end.
# Usage: bash run_probe.sh [NUM_SHARDS]
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR" logs
eval "$(grep 'WANDB_API_KEY' ~/.bashrc)"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda/etc/profile.d/conda.sh
conda activate pirlnav
cd ~/pirlnav

NUM_SHARDS=${1:-4}
pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  python -u run_probe_greedy.py \
      --num-shards "$NUM_SHARDS" --shard-index "$i" \
      --out "logs/probe_greedy_s${i}.json" \
      > "logs/probe_greedy_s${i}.log" 2>&1 &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail -ne 0 ] && echo "WARNING: at least one shard failed; check logs/probe_greedy_s*.log"

echo "=== combined result ==="
python run_probe_greedy.py --merge "logs/probe_greedy_s*.json"
