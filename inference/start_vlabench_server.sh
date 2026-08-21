#!/usr/bin/env bash
# Start FlowWAM inference with the VLABench training/evaluation contract.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to a trained VLABench checkpoint}"
ACTION_NORM_PATH="${ACTION_NORM_PATH:-$(dirname "${CHECKPOINT}")/action_norm_stats.npz}"
LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-${SCRIPT_DIR}/models}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-cuda}"
NUM_FRAMES="${NUM_FRAMES:-33}"
NUM_VIDEO_FRAMES="${NUM_VIDEO_FRAMES:-9}"
VIDEO_INFERENCE_STEPS="${VIDEO_INFERENCE_STEPS:-25}"
ACTION_INFERENCE_STEPS="${ACTION_INFERENCE_STEPS:-50}"
CAMERA_PREFIX="A multi-view video of a Franka robot in T-shape layout: the top row shows the full-size front camera view, the bottom-left shows the half-size left camera view, and the bottom-right shows the half-size wrist camera view. The robot is performing the following task: "

"${PYTHON}" "${SCRIPT_DIR}/flow_action_server.py" \
  --checkpoint "${CHECKPOINT}" \
  --action_norm_path "${ACTION_NORM_PATH}" \
  --local_model_path "${LOCAL_MODEL_PATH}" \
  --checkpoint_mode full \
  --host "${HOST}" --port "${PORT}" --device "${DEVICE}" \
  --action_dim 7 \
  --num_frames "${NUM_FRAMES}" \
  --num_video_frames "${NUM_VIDEO_FRAMES}" \
  --num_action_layers 30 \
  --action_pred_target velocity \
  --action_pos_mode rope \
  --proprio_mode text \
  --cond_layer_stride 1 \
  --action_snr_shift 5.0 \
  --cameras front left wrist \
  --camera_prefix "${CAMERA_PREFIX}" \
  --text_context_dim 4096 \
  --size 320 256 \
  --video_inference_steps "${VIDEO_INFERENCE_STEPS}" \
  --sigma_shift 5.0 \
  --action_inference_steps "${ACTION_INFERENCE_STEPS}" \
  --action_cond_sigma 0.0 \
  --action_chunk_size "${NUM_FRAMES}"
