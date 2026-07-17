# Rental GPU Setup Runbook (Lium, 4x RTX 3090)

Copy-paste runbook to go from a bare rented box to "verified ready for experiments"
without burning billed GPU-hours on environment debugging. Everything here was
link-verified on 2026-07-17 from a machine with no lab-server access — all data sources
below are public EXCEPT the HM3D scenes (need your own Matterport API token).

Good news vs. the old H100 plan (`scripts/setup_h100_env.sh`): RTX 3090 is Ampere
(sm_86), natively supported by the original certified stack (torch 1.12.1+cu113) — no
newer-CUDA rebuild needed. Use the recipe below, not the H100 script.

**Disk requirement: rent a volume with ≥ 150 GB free.** (HM3D scenes ~80GB + demos
+ val episodes + encoder + checkpoints ~200MB each + headroom.)

---

## 0. Have ready before starting the clock

- [ ] SSH details for the box
- [ ] Matterport API token (id + secret) from
      https://matterport.com/habitat-matterport-3d-research-dataset (your own access
      agreement; the only credential-gated download)
- [ ] `WANDB_API_KEY` — **legacy-format** key (new-format keys break `wandb login` on
      this stack; see HANDOVER §6)
- [ ] GitHub access to `KhushiJ2911/pirlnav-costmap` (repo is private — either add the
      box's SSH key as a deploy key, or use a fine-grained PAT for HTTPS clone)

---

## 1. Box sanity check (first 2 minutes)

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
df -h ~ /
free -h
```

Expect 4x RTX 3090 / 24576 MiB and ≥150GB free where you'll work. **Stop and fix
before proceeding if not.** Driver must support CUDA 11.3 runtimes (any driver ≥ 465
does; every 2024+ image qualifies).

If conda is missing on the image:

```bash
curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash && source ~/.bashrc
```

---

## 2. Clone the repo — MUST be at `~/pirlnav` (scripts hardcode `cd ~/pirlnav`)

```bash
cd ~
git clone git@github.com:KhushiJ2911/pirlnav-costmap.git pirlnav
cd ~/pirlnav
git checkout costmap-intern
git submodule update --init --recursive   # takes a few minutes (nested deps)
```

---

## 3. Environment (original certified recipe — works on 3090)

```bash
conda create -n pirlnav python=3.7 cmake=3.14.0 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pirlnav

cd ~/pirlnav/habitat-sim
pip install -r requirements.txt
./build.sh --headless            # long: builds the simulator from source

cd ~/pirlnav/habitat-lab
pip install -r requirements.txt
pip install -e habitat-lab
pip install -e habitat-baselines

cd ~/pirlnav
pip install -e .

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
  --extra-index-url https://download.pytorch.org/whl/cu113

# Extras HANDOVER's env recipe requires:
conda install -c conda-forge lmdb -y
pip install webdataset==0.1.40

# W&B (required: ALL experiments log to project pirlnav-baseline).
# Latest wandb has dropped py3.7 — if the plain install fails, use the pin.
pip install wandb || pip install "wandb==0.15.12"
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
# expect: 1.12.1+cu113 True 4
python -c "import habitat_sim; print(habitat_sim.__version__)"
```

Headless render check (catches missing EGL/GL libs now, not mid-run):

```bash
python -c "
import habitat_sim
cfg = habitat_sim.SimulatorConfiguration()
print('habitat_sim imports and configures OK')"
```

If EGL errors appear: `conda install -c conda-forge libstdcxx-ng libgl libegl -y`.

---

## 4. Required patch (eval breaks without it)

```bash
cd ~/pirlnav
git -C habitat-lab apply ../habitat_lab_eval_fix.patch
```

---

## 5. W&B (scripts grep the key from ~/.bashrc — non-interactive shells skip bashrc)

```bash
echo 'export WANDB_API_KEY=<<FILL IN — legacy-format key>>' >> ~/.bashrc
export WANDB_API_KEY=<<same>>
wandb login "$WANDB_API_KEY"
```

---

## 6. Data (all link-verified 2026-07-17)

Target layout:

```
~/pirlnav/data/
├── scene_datasets/hm3d/                    (§6a — needs Matterport token, ~80GB)
│   └── hm3d_annotated_basis.scene_dataset_config.json   <- configs REQUIRE this file
├── datasets/objectnav/
│   ├── objectnav_hm3d_hd/train/            (§6b — 77k human demos)
│   ├── hm3d/v2/objectnav_hm3d_v2/val/      (§6c — 1000-ep full val)
│   └── overfit_tiny/{train,val}/           (§6d — generated locally)
└── visual_encoders/omnidata_DINO_02.pth    (§6e)
```

### 6a. HM3D scenes (the big one — start it first, in tmux)

```bash
cd ~/pirlnav
python -m habitat_sim.utils.datasets_download --list   # confirm uid names in this habitat-sim version
python -m habitat_sim.utils.datasets_download \
  --username <<MATTERPORT_TOKEN_ID>> --password <<MATTERPORT_TOKEN_SECRET>> \
  --uids hm3d_train_v0.2 hm3d_val_v0.2 \
  --data-path data/
```

The task configs point at `data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json`
— the **annotated** config, which needs the semantic annotation files too. If the uids
above didn't produce that file, also download the semantics uids shown by `--list`
(names vary by habitat-sim version: `hm3d_semantic_annotations_v0.2` /
`hm3d_semantic_configs_v0.2`). **Verify before moving on:**

```bash
ls data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json
```

### 6b. 77k human demos (HF dataset `axel81/pirlnav`, layout verified: `objectnav_hm3d_hd/train/{train.json.gz,content/*.json.gz}`)

```bash
git lfs install
cd ~/pirlnav
git clone https://huggingface.co/datasets/axel81/pirlnav data/datasets/objectnav
ls data/datasets/objectnav/objectnav_hm3d_hd/train/train.json.gz   # must exist
```

(This also brings `objectnav_hm3d_fe/` — frontier-exploration demos, harmless extra.)

### 6c. Full HM3D-v2 val episodes (260MB zip, URL verified live)

```bash
cd ~/pirlnav
mkdir -p data/datasets/objectnav/hm3d/v2
curl -L -o /tmp/objectnav_hm3d_v2.zip \
  https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip
unzip -q /tmp/objectnav_hm3d_v2.zip -d data/datasets/objectnav/hm3d/v2/
ls data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2/val/val.json.gz   # must exist
# if the zip's internal folder nesting differs, mv it so the line above passes
```

### 6d. `overfit_tiny` (generated, not downloaded — needed by run_verify.sh / run_smoke_costmap.sh)

Run AFTER 6b:

```bash
cd ~/pirlnav
python - <<'EOF'
import gzip, json, os
src = "data/datasets/objectnav/objectnav_hm3d_hd/train/train.json.gz"
dst = "data/datasets/objectnav/overfit_tiny/train"
os.makedirs(dst, exist_ok=True)
with gzip.open(src, "rt") as f:
    data = json.load(f)
scene = data["episodes"][0]["scene_id"]
data["episodes"] = [e for e in data["episodes"] if e["scene_id"] == scene][:20]
with gzip.open(f"{dst}/train.json.gz", "wt") as f:
    json.dump(data, f)
print(f"Wrote {len(data['episodes'])} episodes from {scene}")
EOF
mkdir -p data/datasets/objectnav/overfit_tiny/val
cp data/datasets/objectnav/overfit_tiny/train/train.json.gz \
   data/datasets/objectnav/overfit_tiny/val/val.json.gz
```

Note: the demos' `train.json.gz` may keep episodes under per-scene `content/` files
with the top-level file holding only metadata. If the script above writes 0 episodes,
point `src` at one file in `train/content/` instead — same code otherwise.

### 6e. OVRL encoder (official S3 is dead — HF mirror verified, exact filename)

```bash
mkdir -p ~/pirlnav/data/visual_encoders
curl -L -o ~/pirlnav/data/visual_encoders/omnidata_DINO_02.pth \
  https://huggingface.co/gunjan050/ZSON/resolve/main/omnidata_DINO_02.pth
```

---

## 7. Verification gauntlet — all three must pass before real training

```bash
cd ~/pirlnav
mkdir -p tb data/new_checkpoints ~/tmp

# (a) RGB pipeline: overfit 1 scene. PASS = action_ce_loss falls toward 0, accuracy climbs.
#     (script pins CUDA_VISIBLE_DEVICES=2 — fine on a 4-GPU box)
bash run_verify.sh

# (b) Costmap arm smoke: 10 updates end-to-end. PASS = no crash + a log line like
#     "CostmapSensor: built field for (...) in ~7s".
SMOKE_GPU=0 bash run_smoke_costmap.sh

# (c) 30-second eval-path check: run one of the eval scripts against the smoke ckpt
#     just to confirm the patched eval harness loads and steps (Ctrl-C after it starts).
```

Only after (a)+(b) pass is the box certified. Then run the experiments — the full
copy-paste sequence is §9 below.

---

## 9. The experiments — exact command sequence (after §7 passes)

### 9.1 Experiment 1: greedy-descent probe (eval-only, no training cost)

```bash
cd ~/pirlnav
conda activate pirlnav

# (i) cheap API shakeout first — 5 episodes, single process, ~2 min:
python run_probe_greedy.py --max-episodes 5 --out logs/probe_smoke.json

# (ii) full 1000-episode val run, 4 shards in parallel (do this in tmux;
#      several hours if the probe fails episodes at the 500-step cap):
tmux new -s probe
bash run_probe.sh 4
# -> prints combined SR / SPL / softSPL at the end; per-shard JSONs in logs/
```

**Decision gate:** high SR → input is sufficient, teacher was the bottleneck →
proceed to 9.2. Low SR → inspect failures before training anything (FOV / depth
window / 64x64 may be insufficient — see PROJECT_CONTEXT.md). Optional knob:
`--stop-threshold` (default 0.25m).

### 9.2 Experiment 2 step 1: generate shortest-path demos (CPU-bound, ~hours)

```bash
cd ~/pirlnav
tmux new -s gensp
bash run_gen_sp_demos.sh 8
# safe to re-run after interruption — finished scenes are skipped.
# FINAL LINE prints:  suggested TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF = <X>
# SAVE THAT NUMBER — the training scripts require it.
```

### 9.3 Experiment 2 step 2: train both arms in parallel (1 GPU-day each)

```bash
cd ~/pirlnav
COEF=<X-from-9.2>

tmux new -s train_rgb_sp -d \
  "INFLECTION_COEF=$COEF TRAIN_GPU=0 bash run_train_1gpu_sp.sh"
tmux new -s train_cm_sp -d \
  "INFLECTION_COEF=$COEF TRAIN_GPU=1 bash run_train_costmap_1gpu_sp.sh"

# watch: tmux attach -t train_cm_sp ; W&B runs il_rgb_sp_10env_acc51 /
# il_costmap_2ch_sp_10env_acc51 in project pirlnav-baseline.
# Reminder: losses/* are the only meaningful train metrics (teacher forcing).
```

### 9.4 Experiment 2 step 3: eval both arms on full val (can start when ckpt.1 exists)

```bash
cd ~/pirlnav
tmux new -s eval_rgb_sp -d "EVAL_GPU=2 bash run_eval_sp.sh"
tmux new -s eval_cm_sp  -d "EVAL_GPU=3 bash run_eval_costmap_sp.sh"
# Both skip checkpoints that don't exist yet — re-run them after training
# finishes to pick up the remaining ckpts. Results land on W&B
# (il_rgb_sp_eval / il_costmap_2ch_sp_eval).
```

**The headline readout:** costmap-SP eval SR vs. RGB-SP eval SR vs. the two
human-demo arms (both 0.000). Prediction (PROJECT_CONTEXT.md): costmap-SP high,
RGB-SP low — the interaction effect.

---

## 10. 4x 3090 usage notes

- The certified configs are 1-GPU. Simplest use of 4 GPUs: run independent jobs in
  parallel (e.g. exp-2's RGB arm on GPU 0, costmap arm on GPU 1, evals on 2/3), each
  with its own `CUDA_VISIBLE_DEVICES` and **distinct MAIN_PORT** (see HANDOVER §7).
- 24GB > the 16GB the configs were sized for — you can raise NUM_ENVIRONMENTS and cut
  NUM_ACCUM_STEPS to keep effective batch = envs x 32 x accum ≈ 16,320. Retune ONLY
  after §7 passes, and change one thing at a time.
- Costmap arm is CPU-hungry (sim + field builds): check CPU count; on weak-CPU rentals
  keep NUM_ENVIRONMENTS at 10.
