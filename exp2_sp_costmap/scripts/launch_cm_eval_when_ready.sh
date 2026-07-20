#!/bin/bash
export LC_ALL=C
cd ~/pirlnav
# wait for val field precompute to finish
for i in $(seq 1 120); do
  grep -q "DONE. counts" ~/precompute_val.log 2>/dev/null && break
  sleep 15
done
sleep 3
echo "val precompute done: $(grep 'DONE. counts' ~/precompute_val.log)" >> ~/eval_costmap_launcher.log
echo "val fields in cache: $(ls data/costmap_field_cache/*.npz 2>/dev/null | wc -l)" >> ~/eval_costmap_launcher.log
tmux new-session -d -s evalcm "bash ~/run_eval_costmap_gpu0.sh > ~/eval_costmap.log 2>&1"
echo "COSTMAP EVAL LAUNCHED @ $(date)" >> ~/eval_costmap_launcher.log
