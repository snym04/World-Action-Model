

# FlowWAM: Optical Flow as a Unified Action Representation for World Action Models

[🏠 Project Page](https://flow-wam.github.io/)  | 
[📄 Paper](https://arxiv.org/abs/XXXX.XXXXX)  | 
[🤗 Checkpoints](https://huggingface.co/YixiangChen/FlowWAM)  | 
[📊 Dataset](https://huggingface.co/datasets/YixiangChen/FlowWAM_RoboTwin)



**TL;DR:** World Action Models repurpose pretrained video generators for control, yet a modality gap persists: action signals must conform to the generator's visual priors while preserving the dense cross-frame motion that control requires. FlowWAM closes this gap with optical flow, a unified, video-native action representation that shares the RGB format, encodes spatially grounded per-pixel displacement, and is extractable from action-unlabeled video. A single shared dual-stream diffusion model generates flow for action prediction, conditions on flow for world modeling, and pretrains on large-scale unlabeled video.

## Contents

- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Inference (RoboTwin Evaluation)](#inference-robotwin-evaluation)
- [Training](#training)
- [Citation](#citation)



## Repository Layout

```text
FlowWAM/
├── diffsynth/                       # Trimmed diffusion library (Wan2.2 + dual-stream)
├── training/                        # Training code and launchers
│   ├── train.sh                     # Training launcher
│   ├── precompute.sh                # Optional latent-cache launcher
│   ├── flow_action_train.py         # Training entry (module + loop)
│   ├── action_dit.py                # IDM action expert
│   └── dataset_action_robotwin.py   # RoboTwin RGB + flow + action dataset
├── inference/                       # Inference server + RoboTwin policy
│   ├── start_server.sh              # Server launcher
│   ├── flow_action_server.py        # WebSocket inference server
│   └── robotwin_policy/             # Drop-in RoboTwin policy (→ policy/flowwam)
│       ├── eval.sh                  # Eval one task
│       └── eval_all.sh              # Eval all 50 tasks (multi-GPU aware)
└── data_generation/                 # Robot-only renderer (RoboTwin data patch)
    ├── script/render_robot_only.py           # → RoboTwin: script/
    └── envs/utils/robot_only_renderer.py     # → RoboTwin: envs/utils/
```



## Installation

```bash
conda create -n flowwam python=3.10 -y
conda activate flowwam
pip install -U pip

# PyTorch + torchvision matching your CUDA (torchvision provides RAFT).
# Tested with torch 2.5.1 / torchvision 0.20.1 on CUDA 12.4.
pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchvision==0.20.1
pip install -r requirements.txt
pip install -e .
```

This `flowwam` env runs **training and the inference server**. Evaluation also
needs a **second, separate RoboTwin conda environment** — the simulator and its
deps (SAPIEN, etc.) come from the official
[RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin), not from
FlowWAM. Into that RoboTwin env, add only the thin client's deps:

```bash
conda activate RoboTwin        # your RoboTwin env
pip install websockets msgpack numpy
```

## Inference (RoboTwin Evaluation)

### 1. Download the base Wan models

FlowWAM runs on the `Wan2.2-TI2V-5B` backbone (DiT + VAE + T5); the T5 tokenizer
comes from `Wan2.1-T2V-1.3B`.

```bash
pip install -U "huggingface_hub[cli]"

hf download Wan-AI/Wan2.2-TI2V-5B \
    --local-dir training/models/Wan-AI/Wan2.2-TI2V-5B
hf download Wan-AI/Wan2.1-T2V-1.3B --include "google/*" \
    --local-dir training/models/Wan-AI/Wan2.1-T2V-1.3B

# The server reads base weights from inference/models — link them once.
mkdir -p inference/models
ln -sfn ../../training/models/Wan-AI inference/models/Wan-AI
```



### 2. Download the FlowWAM checkpoint

```bash
hf download YixiangChen/FlowWAM \
    flowwam_robotwin.safetensors \
    flowwam_robotwin_action_norm_stats.npz \
    --local-dir ./checkpoints
```



### 3. Start the server (`flowwam` env, GPU)

```bash
CHECKPOINT=checkpoints/flowwam_robotwin.safetensors \
ACTION_NORM_PATH=checkpoints/flowwam_robotwin_action_norm_stats.npz \
bash inference/start_server.sh
```

Wait until it reports it is serving on `ws://0.0.0.0:8000`.

### 4. Run evaluation (RoboTwin env)

Activate your RoboTwin conda env first. FlowWAM plugs in as an **additive policy
package** — RoboTwin auto-discovers policies under `policy/<name>/`, so this repo's
[`inference/robotwin_policy/`](inference/robotwin_policy/) is linked in as
`policy/flowwam` and driven by RoboTwin's stock `script/eval_policy.py` (**no
RoboTwin source file is edited**). `eval.sh` creates the link for you; to do it
manually:

```bash
ln -sfn "$(pwd)/inference/robotwin_policy" "${ROBOTWIN_ROOT}/policy/flowwam"
```

Then run evaluation:
```bash
# One task:  <task_name> <task_config> <seed> <gpu_id>
ROBOTWIN_ROOT=/path/to/RoboTwin \
    bash inference/robotwin_policy/eval.sh place_dual_shoes demo_clean 0 0

# All 50 tasks (task_config defaults to demo_clean):
ROBOTWIN_ROOT=/path/to/RoboTwin \
    bash inference/robotwin_policy/eval_all.sh demo_clean
```

Results are written to `${ROBOTWIN_ROOT}/eval_result/<task>/flowwam/<task_config>/`.
`task_config` is the RoboTwin scene config (`demo_clean` or `demo_randomized`).

**Multi-GPU.** Start one server per GPU (ports `8000, 8001, …`), then let
`eval_all.sh` shard the 50 tasks across them:

```bash
# Server env — one server per GPU (e.g. 4 GPUs):
for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g PORT=$((8000+g)) \
    CHECKPOINT=checkpoints/flowwam_robotwin.safetensors \
    ACTION_NORM_PATH=checkpoints/flowwam_robotwin_action_norm_stats.npz \
    bash inference/start_server.sh &
done

# RoboTwin env — tasks are dispatched across the 4 servers automatically:
ROBOTWIN_ROOT=/path/to/RoboTwin NUM_GPUS=4 \
    bash inference/robotwin_policy/eval_all.sh demo_clean
```



## Training

### 1. Get the data

FlowWAM data is standard RoboTwin 2.0 `aloha-agilex` demonstrations plus **one
addition**: a robot-only view of each episode (the arm rendered alone in an empty
scene), used to compute optical flow. Each task/variant follows:

```text
<data_root>/<task>/<variant>/
├── data/episode*.hdf5              # scene RGB + joint_action        (standard RoboTwin)
├── robot_only/data/episode*.hdf5   # robot-only RGB for optical flow (FlowWAM addition)
└── instructions/episode*.json      # language instructions           (standard RoboTwin)
```

Get it either way:

**A) Download the ready-made dataset** (robot-only already included):

```bash
DATA_ROOT=/path/to/robotwin_data
hf download YixiangChen/FlowWAM_RoboTwin --repo-type dataset --local-dir /tmp/flowwam_dl
mkdir -p "$DATA_ROOT"
for f in /tmp/flowwam_dl/*/*.tar.gz; do tar -xzf "$f" -C "$DATA_ROOT"; done
# Each archive extracts to <data_root>/<task>/<variant>/. Grab a subset by
# downloading only specific aloha-agilex_clean_50/<task>.tar.gz paths.
```

**B) Generate from your own RoboTwin data.** If you already have standard RoboTwin
demonstrations, add only the missing `robot_only/` view with the renderer in
`[data_generation/](data_generation/)` (drop it into your RoboTwin checkout):

```bash
# In your RoboTwin checkout; 2nd arg is the <variant> folder holding data/:
python script/render_robot_only.py place_dual_shoes aloha-agilex_clean_50
```

The two default variants are `aloha-agilex_clean_50` (50 episodes/task) and
`aloha-agilex_randomized_500` (500 episodes/task); override with `VARIANTS`. Set
`FLOW_MODE=full_scene` to compute flow on the full scene and skip `robot_only/`.

### 2. Download the base Wan models

Same as [Inference step 1](#1-download-the-base-wan-models) — the
`Wan2.2-TI2V-5B` + `Wan2.1-T2V-1.3B` weights under `training/models/`.

### 3. Train

On the first run, action-normalization stats are computed and saved to
`action_norm_stats.npz` in the output directory
(`training/models/train/flowwam_idm/`), next to the checkpoints.

```bash
cd training
export SWANLAB_API_KEY=...        # optional: SwanLab logging

# Single node, all visible GPUs (online RAFT flow).
DATASET_BASE_PATH=/path/to/robotwin_data bash train.sh

# Multi node: run on every node with matching NUM_MACHINES / MASTER_ADDR.
NUM_MACHINES=2 MASTER_ADDR=<ip> bash train.sh --machine_rank 0   # master
NUM_MACHINES=2 MASTER_ADDR=<ip> bash train.sh --machine_rank 1   # worker
```

Useful env vars: `VARIANTS` (subset of variants), `RESUME_CHECKPOINT` (warm-start
the video DiT + flow_stream), `RESUME_STATE_DIR=.../state/step-N` (resume exactly),
`LOAD_FROM_CACHE=true CACHE_ROOT=...` (train from a precomputed cache built by
`bash training/precompute.sh`). Checkpoints (`step-XXXX.safetensors`) are written
every `SAVE_STEPS` (default 2000). All knobs are documented at the top of
`[training/train.sh](training/train.sh)`.

To evaluate your own checkpoint, point the server at it in
[Inference step 3](#3-start-the-server-flowwam-env-gpu):

```bash
CHECKPOINT=training/models/train/flowwam_idm/step-XXXX.safetensors \
bash inference/start_server.sh   # action_norm_stats.npz is picked up from the same dir
```



## Citation

```bibtex
@article{flowwam,
  title   = {FlowWAM: A Dual-Stream RGB + Optical-Flow World Action Model for RoboTwin},
  author  = {TODO},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026},
  url     = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

