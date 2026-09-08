#!/usr/bin/env bash
# FlowWAM training on the official VLABench 10-task primitive fine-tuning set.
# Benchmark-boundary changes only: raw HDF5 adapter, 7D absolute EE actions,
# and VLABench cameras. The original FlowWAM temporal, flow, IDM, optimizer,
# loss, and checkpointing logic is intentionally preserved.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
: "${SWANLAB_API_KEY:?SWANLAB_API_KEY must be provided by the WAM secret gate}"

# ---- Distributed config ----
NUM_MACHINES="${NUM_MACHINES:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
MACHINE_RANK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --machine_rank) MACHINE_RANK="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"

# ---- Official VLABench primitive fine-tuning set ----
DATASET_BASE_PATH="${DATASET_BASE_PATH:?set DATASET_BASE_PATH to the extracted VLABench root}"
TASK_NAMES=(
  add_condiment insert_flower select_book select_chemistry_tube select_drink
  select_fruit select_mahjong select_painting select_poker select_toy
)
CAMERAS=(front left wrist)
GRIPPER_OPEN_THRESHOLD="${GRIPPER_OPEN_THRESHOLD:-0.03}"

# ---- Base models: same Wan2.2-TI2V-5B components as upstream FlowWAM ----
MODEL_PATHS_DIT="Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors"
MODEL_PATHS_T5="Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth"
MODEL_PATHS_VAE="Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"
MODEL_PATHS="${MODEL_PATHS_DIT},${MODEL_PATHS_T5},${MODEL_PATHS_VAE}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:?set MODEL_CACHE_ROOT to the shared models directory}"
export TORCH_HOME="${TORCH_HOME:-${MODEL_CACHE_ROOT}/torch}"
MODEL_LINK="${SCRIPT_DIR}/models"
if [[ -L "${MODEL_LINK}" ]]; then
  if [[ "$(readlink -f "${MODEL_LINK}")" != "$(readlink -f "${MODEL_CACHE_ROOT}")" ]]; then
    echo "${MODEL_LINK} points to a different model cache" >&2
    exit 1
  fi
elif [[ -e "${MODEL_LINK}" ]]; then
  echo "${MODEL_LINK} exists but is not the manifest-selected cache link" >&2
  exit 1
else
  ln -s "${MODEL_CACHE_ROOT}" "${MODEL_LINK}"
fi
for required in \
  "Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth" \
  "Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth" \
  "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json" \
  "torch/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth"; do
  if [[ ! -f "${MODEL_CACHE_ROOT}/${required}" ]]; then
    echo "Missing required model file: ${MODEL_CACHE_ROOT}/${required}" >&2
    exit 1
  fi
done
if ! compgen -G "${MODEL_CACHE_ROOT}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model*.safetensors" >/dev/null; then
  echo "Missing Wan2.2 DiT safetensors under ${MODEL_CACHE_ROOT}" >&2
  exit 1
fi

# ---- Original FlowWAM temporal contract: 1 anchor + 32 actions ----
NUM_FRAMES="${NUM_FRAMES:-33}"
NUM_VIDEO_FRAMES="${NUM_VIDEO_FRAMES:-9}"
VISUAL_STRIDE="${VISUAL_STRIDE:-4}"
SIZE_W=320
SIZE_H=256

# ---- Original robot-only optical-flow path ----
FLOW_METHOD=raft
FLOW_DEVICE=cuda
FLOW_MODE="${FLOW_MODE:-robot_only}"
FLOW_MAX_MAGNITUDE="${FLOW_MAX_MAGNITUDE:-25.0}"
FLOW_MOTION_BOOST=2.0

# ---- Original FlowWAM optimization / IDM settings ----
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
TRAINABLE_MODELS=dit
ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-1.0}"
FLOW_LOSS_WEIGHT=0.1
ACTION_DIM=7
NUM_ACTION_LAYERS=30
ACTION_SNR_SHIFT="${ACTION_SNR_SHIFT:-5.0}"
TEXT_CONTEXT_DIM=4096
LOSS_TIMESTEP_WEIGHTING=on
REF_AUG_STRENGTH=0.1
COND_NOISE_PROB="${COND_NOISE_PROB:-0.5}"
COND_DETACH="${COND_DETACH:-false}"
COND_LAYER_STRIDE=1
ACTION_PRED_TARGET=velocity
ACTION_POS_MODE=rope
PROPRIO_MODE=text
LR_SCHEDULER_TYPE=cosine
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-1000}"
LR_MAX_STEPS="${LR_MAX_STEPS:-0}"

# ---- Reproducible outputs and resume ----
SEED="${SEED:-42}"
OUTPUT_PATH="${OUTPUT_PATH:?set OUTPUT_PATH to the manifest-bound run directory}"
ACTION_NORM_PATH="${OUTPUT_PATH}/action_norm_stats.npz"
SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_EVERY_N_EPOCHS=100
FULL_STATE_KEEP=2
RESUME_STATE_DIR="${RESUME_STATE_DIR:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s ',' 0 $((NUM_GPUS - 1)))}"
ACCELERATE_ARGS="--num_processes=$((NUM_GPUS * NUM_MACHINES)) --num_machines=${NUM_MACHINES}"
if [[ "${NUM_MACHINES}" -gt 1 ]]; then
  ACCELERATE_ARGS="${ACCELERATE_ARGS} --machine_rank=${MACHINE_RANK} --main_process_ip=${MASTER_ADDR} --main_process_port=${MASTER_PORT}"
fi

echo "========== FlowWAM VLABench Training =========="
echo "  machines/gpus:   ${NUM_MACHINES} x ${NUM_GPUS}"
echo "  frames:          action=${NUM_FRAMES} video=${NUM_VIDEO_FRAMES} stride=${VISUAL_STRIDE}"
echo "  batch/lr:        ${BATCH_SIZE} / ${LEARNING_RATE}"
echo "  seed:            ${SEED}"
echo "  tasks/cameras:   ${#TASK_NAMES[@]} / ${CAMERAS[*]}"
echo "  output:          ${OUTPUT_PATH}"
echo "================================================"

cd "${SCRIPT_DIR}"
"${PYTHON}" -m accelerate.commands.launch \
  ${ACCELERATE_ARGS} \
  "${SCRIPT_DIR}/flow_action_train.py" \
  --dataset_type vlabench \
  --seed "${SEED}" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --task_names "${TASK_NAMES[@]}" \
  --num_frames "${NUM_FRAMES}" \
  --num_video_frames "${NUM_VIDEO_FRAMES}" \
  --visual_stride "${VISUAL_STRIDE}" \
  --model_id_with_origin_paths "${MODEL_PATHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --max_train_steps "${MAX_TRAIN_STEPS}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --size "${SIZE_W}" "${SIZE_H}" \
  --cameras "${CAMERAS[@]}" \
  --trainable_models "${TRAINABLE_MODELS}" \
  --flow_method "${FLOW_METHOD}" \
  --flow_device "${FLOW_DEVICE}" \
  --flow_mode "${FLOW_MODE}" \
  --flow_max_magnitude "${FLOW_MAX_MAGNITUDE}" \
  --dataset_num_workers 0 \
  --flow_loss_weight "${FLOW_LOSS_WEIGHT}" \
  --action_loss_weight "${ACTION_LOSS_WEIGHT}" \
  --action_dim "${ACTION_DIM}" \
  --num_action_layers "${NUM_ACTION_LAYERS}" \
  --action_norm_path "${ACTION_NORM_PATH}" \
  --gripper_open_threshold "${GRIPPER_OPEN_THRESHOLD}" \
  --action_snr_shift "${ACTION_SNR_SHIFT}" \
  --cond_noise_prob "${COND_NOISE_PROB}" \
  $([[ "${COND_DETACH}" = "true" ]] && echo "--cond_detach") \
  --cond_layer_stride "${COND_LAYER_STRIDE}" \
  --action_pred_target "${ACTION_PRED_TARGET}" \
  --action_pos_mode "${ACTION_POS_MODE}" \
  --proprio_mode "${PROPRIO_MODE}" \
  --loss_timestep_weighting "${LOSS_TIMESTEP_WEIGHTING}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --lr_warmup_steps "${LR_WARMUP_STEPS}" \
  --lr_max_steps "${LR_MAX_STEPS}" \
  --full_state_keep "${FULL_STATE_KEEP}" \
  ${RESUME_STATE_DIR:+--resume_state_dir "${RESUME_STATE_DIR}"} \
  --flow_motion_boost "${FLOW_MOTION_BOOST}" \
  --text_context_dim "${TEXT_CONTEXT_DIM}" \
  --batch_size "${BATCH_SIZE}" \
  --extra_inputs "input_image" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --ref_aug_strength "${REF_AUG_STRENGTH}" \
  --save_every_n_epochs "${SAVE_EVERY_N_EPOCHS}" \
  ${SAVE_STEPS:+--save_steps "${SAVE_STEPS}"} \
  --fp32_modulation \
  --use_gradient_checkpointing \
  ${RESUME_CHECKPOINT:+--resume_checkpoint "${RESUME_CHECKPOINT}"}
