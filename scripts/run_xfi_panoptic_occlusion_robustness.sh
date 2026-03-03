#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_CFG="${TRAIN_CFG:-${REPO_ROOT}/configs/baseline/occlusion_robustness/xfi_panoptic_train_temporal_fg_rgb_anchor.yml}"
TEST_CFG_OCC="${TEST_CFG_OCC:-${REPO_ROOT}/configs/baseline/occlusion_robustness/xfi_panoptic_test_occluded.yml}"
TEST_CFG_UNOCC="${TEST_CFG_UNOCC:-${REPO_ROOT}/configs/baseline/occlusion_robustness/xfi_panoptic_test_unoccluded.yml}"

GPUS="${GPUS:-1}"
NUM_NODES="${NUM_NODES:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BATCH_SIZE_EVA="${BATCH_SIZE_EVA:-32}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PIN_MEMORY="${PIN_MEMORY:-1}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_OFFLINE="${WANDB_OFFLINE:-0}"
EXP_NAME="${EXP_NAME:-occlusion_robustness_xfi}"
VERSION="${VERSION:-$(date +'%Y%m%d_%H%M%S')_train}"

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

echo "[xfi-occlusion] training config: ${TRAIN_CFG}"
uv run python "${REPO_ROOT}/main.py" \
  -c "${TRAIN_CFG}" \
  "${MAIN_ARGS[@]}" \
  --exp_name "${EXP_NAME}" \
  --version "${VERSION}"

CKPT_DIR="${REPO_ROOT}/logs/${EXP_NAME}/${VERSION}"
BEST_CKPT="$(find "${CKPT_DIR}" -maxdepth 1 -type f -name '*.ckpt' ! -name 'last.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | awk '{print $2}')"
if [[ -z "${BEST_CKPT}" ]]; then
  if [[ -f "${CKPT_DIR}/last.ckpt" ]]; then
    BEST_CKPT="${CKPT_DIR}/last.ckpt"
  fi
fi
if [[ -z "${BEST_CKPT}" || ! -f "${BEST_CKPT}" ]]; then
  echo "[xfi-occlusion] ERROR: no checkpoint found under ${CKPT_DIR}" >&2
  exit 1
fi

echo "[xfi-occlusion] testing checkpoint: ${BEST_CKPT}"

TEST_VERSION_OCC="${VERSION}_test_occluded"
uv run python "${REPO_ROOT}/main.py" \
  -c "${TEST_CFG_OCC}" \
  "${MAIN_ARGS[@]}" \
  --test \
  --checkpoint_path "${BEST_CKPT}" \
  --exp_name "${EXP_NAME}" \
  --version "${TEST_VERSION_OCC}"

TEST_VERSION_UNOCC="${VERSION}_test_unoccluded"
uv run python "${REPO_ROOT}/main.py" \
  -c "${TEST_CFG_UNOCC}" \
  "${MAIN_ARGS[@]}" \
  --test \
  --checkpoint_path "${BEST_CKPT}" \
  --exp_name "${EXP_NAME}" \
  --version "${TEST_VERSION_UNOCC}"

echo "[xfi-occlusion] done"
