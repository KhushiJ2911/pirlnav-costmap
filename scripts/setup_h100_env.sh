#!/usr/bin/env bash
set -euo pipefail

# H100-safe setup template for this repo.
# This is intentionally conservative and uses a fresh conda env.
# It is a build recipe, not a "press run and it will work everywhere" script.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ENV_NAME:-pirlnav_h100}"
PYTHON_VER="${PYTHON_VER:-3.10}"
CUDA_WHEEL="${CUDA_WHEEL:-cu121}"

# NOTE:
# - The repo README's original stack (py3.7 + torch 1.12.1+cu113) is not a safe H100 starting point.
# - The H100 path should use a newer torch/CUDA stack and a source build of habitat-sim.
# - If you are missing GPU drivers or container runtime support, this will still fail before training,
#   so verify the machine is actually exposing a CUDA device first.

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not on PATH."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found. This host may not expose the GPU driver correctly."
  echo "         Do not start long training until this command works on the H100 host."
fi

# 1) Create fresh conda env.
conda create -n "$ENV_NAME" python="$PYTHON_VER" -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 2) Base tools.
pip install -U pip setuptools wheel cmake ninja

# 3) Install a CUDA-compatible torch stack. This is the most important compatibility point.
#    For H100, use a modern CUDA wheel. If your host's CUDA driver is 12.1+, cu121 is the safe start.
pip install \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/"$CUDA_WHEEL"

# 4) Install habitat-sim dependencies from the repo.
pip install -r "$REPO_ROOT/habitat-sim/requirements.txt"

# 5) Build habitat-sim from source in headless mode.
#    This is required because the repo's stack expects a local source build, not a pip wheel only.
cd "$REPO_ROOT/habitat-sim"
./build.sh --headless

# 6) Install habitat-lab requirements.
cd "$REPO_ROOT/habitat-lab"
pip install -r requirements.txt
pip install -e habitat-lab
pip install -e habitat-baselines

# 7) Install the local project package.
cd "$REPO_ROOT"
pip install -e .

# 8) Verify that the required datasets and encoder exist.
for path in \
  "$REPO_ROOT/data/scene_datasets/hm3d" \
  "$REPO_ROOT/data/datasets/objectnav/objectnav_hm3d_hd" \
  "$REPO_ROOT/data/visual_encoders/omnidata_DINO_02.pth"
do
  if [ ! -e "$path" ]; then
    echo "WARNING: expected path not found: $path"
    echo "         You may need to mount or symlink the dataset and encoder from the correct storage location."
  fi
done

# 9) Smoke test the costmap path before spending a long H100 billing cycle.
#    This should be the first thing you run after the environment is built.
#    You should use this repo's smoke script, not a full training job.
cd "$REPO_ROOT"
echo ""
echo "H100 environment setup complete."
echo "Next: run the smoke test from the repo with:"
echo "  bash run_smoke_costmap.sh"
echo ""
