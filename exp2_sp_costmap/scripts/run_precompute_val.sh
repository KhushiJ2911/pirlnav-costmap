#!/bin/bash
export LC_ALL=C
cd ~/pirlnav
~/miniconda3/envs/pirlnav/bin/python /root/precompute_fields.py \
  "data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2/val/content/*.json.gz" \
  data/costmap_field_cache
echo "VAL-PRECOMPUTE-EXIT-$?"
