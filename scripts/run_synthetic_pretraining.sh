#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPUS="${GPUS:-2}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0,1}"
NUM_NODES="${NUM_NODES:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BATCH_SIZE_EVA="${BATCH_SIZE_EVA:-32}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PIN_MEMORY="${PIN_MEMORY:-1}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_OFFLINE="${WANDB_OFFLINE:-0}"

WAIT_FOR_FREE_GPUS="${WAIT_FOR_FREE_GPUS:-1}"
FREE_GPU_MAX_UTIL="${FREE_GPU_MAX_UTIL:-20}"
FREE_GPU_MAX_MEM_MB="${FREE_GPU_MAX_MEM_MB:-2048}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"

RUN_ID="${RUN_ID:-$(date +'%Y%m%d_%H%M%S')}"
VERSION_PREFIX="${VERSION_PREFIX:-pretrain}"
HUMMAN_STAGE1_EXP_NAME="${HUMMAN_STAGE1_EXP_NAME:-synthetic_transfer_humman_stage1}"
HUMMAN_STAGE2_EXP_NAME="${HUMMAN_STAGE2_EXP_NAME:-synthetic_transfer_humman_stage2}"
PANOPTIC_STAGE1_EXP_NAME="${PANOPTIC_STAGE1_EXP_NAME:-synthetic_transfer_panoptic_stage1}"
PANOPTIC_STAGE2_EXP_NAME="${PANOPTIC_STAGE2_EXP_NAME:-synthetic_transfer_panoptic_stage2}"
HUMMAN_STAGE1_VERSION="${HUMMAN_STAGE1_VERSION:-${VERSION_PREFIX}_${RUN_ID}}"
HUMMAN_STAGE2_VERSION="${HUMMAN_STAGE2_VERSION:-${VERSION_PREFIX}_${RUN_ID}}"
PANOPTIC_STAGE1_VERSION="${PANOPTIC_STAGE1_VERSION:-${VERSION_PREFIX}_${RUN_ID}}"
PANOPTIC_STAGE2_VERSION="${PANOPTIC_STAGE2_VERSION:-${VERSION_PREFIX}_${RUN_ID}}"

HUMMAN_STAGE1_CFG="${HUMMAN_STAGE1_CFG:-${REPO_ROOT}/configs/exp/synthetic_transfer/humman/synthetic_stage1_pretrain.yml}"
HUMMAN_STAGE2_CFG="${HUMMAN_STAGE2_CFG:-${REPO_ROOT}/configs/exp/synthetic_transfer/humman/synthetic_stage2_pretrain.yml}"
PANOPTIC_STAGE1_CFG="${PANOPTIC_STAGE1_CFG:-${REPO_ROOT}/configs/exp/synthetic_transfer/panoptic/synthetic_stage1_pretrain.yml}"
PANOPTIC_STAGE2_CFG="${PANOPTIC_STAGE2_CFG:-${REPO_ROOT}/configs/exp/synthetic_transfer/panoptic/synthetic_stage2_pretrain.yml}"

RUN_LOG_DIR="${RUN_LOG_DIR:-${REPO_ROOT}/logs/synthetic_pretraining_runs/${RUN_ID}}"
mkdir -p "${RUN_LOG_DIR}"

MAIN_ARGS=(
  -g "${GPUS}"
  -n "${NUM_NODES}"
  -w "${NUM_WORKERS}"
  -b "${BATCH_SIZE}"
  -e "${BATCH_SIZE_EVA}"
  -p "${PREFETCH_FACTOR}"
)
if [[ "${PIN_MEMORY}" == "1" ]]; then
  MAIN_ARGS+=(--pin_memory)
fi
if [[ "${USE_WANDB}" == "1" ]]; then
  MAIN_ARGS+=(--use_wandb)
  if [[ "${WANDB_OFFLINE}" == "1" ]]; then
    MAIN_ARGS+=(--wandb_offline)
  fi
fi
MAIN_ARGS+=("$@")

log() {
  printf '[synthetic-pretrain] %s\n' "$*"
}

find_best_ckpt() {
  local ckpt_dir="$1"
  local best_ckpt
  best_ckpt="$(find "${ckpt_dir}" -maxdepth 1 -type f -name '*.ckpt' ! -name 'last.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | awk '{print $2}')"
  if [[ -z "${best_ckpt}" && -f "${ckpt_dir}/last.ckpt" ]]; then
    best_ckpt="${ckpt_dir}/last.ckpt"
  fi
  if [[ -z "${best_ckpt}" || ! -f "${best_ckpt}" ]]; then
    log "ERROR: no checkpoint found under ${ckpt_dir}" >&2
    return 1
  fi
  printf '%s\n' "${best_ckpt}"
}

wait_for_free_gpus() {
  if [[ "${WAIT_FOR_FREE_GPUS}" != "1" ]]; then
    return 0
  fi
  while true; do
    local query
    query="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
    if awk -F',' -v max_mem="${FREE_GPU_MAX_MEM_MB}" -v max_util="${FREE_GPU_MAX_UTIL}" '
      BEGIN { ok = 1 }
      {
        gsub(/ /, "", $1)
        gsub(/ /, "", $2)
        if (($1 + 0) > max_mem || ($2 + 0) > max_util) {
          ok = 0
        }
      }
      END { exit ok ? 0 : 1 }
    ' <<<"${query}"; then
      log "GPU wait condition satisfied."
      return 0
    fi
    log "GPUs still busy. Waiting ${GPU_POLL_SECONDS}s. Thresholds: mem<=${FREE_GPU_MAX_MEM_MB}MB util<=${FREE_GPU_MAX_UTIL}%."
    sleep "${GPU_POLL_SECONDS}"
  done
}

run_stage() {
  local stage_name="$1"
  local cfg_path="$2"
  local exp_name="$3"
  local version="$4"
  local checkpoint_path="${5:-}"
  local log_path="${RUN_LOG_DIR}/${stage_name}.log"
  local cmd=(
    uv run python "${REPO_ROOT}/main.py"
    -c "${cfg_path}"
    "${MAIN_ARGS[@]}"
    --exp_name "${exp_name}"
    --version "${version}"
  )
  if [[ -n "${checkpoint_path}" ]]; then
    cmd+=(--checkpoint_path "${checkpoint_path}")
  fi

  log "Starting ${stage_name}: cfg=${cfg_path} exp=${exp_name} version=${version}"
  (
    cd "${REPO_ROOT}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" "${cmd[@]}"
  ) 2>&1 | tee "${log_path}"
}

wait_for_free_gpus

run_stage \
  "humman_stage1" \
  "${HUMMAN_STAGE1_CFG}" \
  "${HUMMAN_STAGE1_EXP_NAME}" \
  "${HUMMAN_STAGE1_VERSION}"

HUMMAN_STAGE1_CKPT="$(find_best_ckpt "${REPO_ROOT}/logs/${HUMMAN_STAGE1_EXP_NAME}/${HUMMAN_STAGE1_VERSION}")"
log "HuMMan stage-1 checkpoint: ${HUMMAN_STAGE1_CKPT}"

run_stage \
  "humman_stage2" \
  "${HUMMAN_STAGE2_CFG}" \
  "${HUMMAN_STAGE2_EXP_NAME}" \
  "${HUMMAN_STAGE2_VERSION}" \
  "${HUMMAN_STAGE1_CKPT}"

run_stage \
  "panoptic_stage1" \
  "${PANOPTIC_STAGE1_CFG}" \
  "${PANOPTIC_STAGE1_EXP_NAME}" \
  "${PANOPTIC_STAGE1_VERSION}"

PANOPTIC_STAGE1_CKPT="$(find_best_ckpt "${REPO_ROOT}/logs/${PANOPTIC_STAGE1_EXP_NAME}/${PANOPTIC_STAGE1_VERSION}")"
log "Panoptic stage-1 checkpoint: ${PANOPTIC_STAGE1_CKPT}"

run_stage \
  "panoptic_stage2" \
  "${PANOPTIC_STAGE2_CFG}" \
  "${PANOPTIC_STAGE2_EXP_NAME}" \
  "${PANOPTIC_STAGE2_VERSION}" \
  "${PANOPTIC_STAGE1_CKPT}"

log "Synthetic pretraining sequence finished."
