#!/bin/bash
export LC_ALL=C
cd ~/pirlnav
~/miniconda3/envs/pirlnav/bin/python /root/precompute_fields.py \
  "data/datasets/objectnav/objectnav_hm3d_sp_gen/train/content/*.json.gz" \
  data/costmap_field_cache
echo "PRECOMPUTE-EXIT-$?"
