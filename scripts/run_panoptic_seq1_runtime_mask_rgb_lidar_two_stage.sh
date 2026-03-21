#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

STAGE1_TRAIN_CFG="${STAGE1_TRAIN_CFG:-${REPO_ROOT}/configs/baseline/occlusion_robustness/panoptic_seq1_runtime_mask_rgb_lidar/panoptic_seq1_panoptic_train_temporal_fg_runtime_mask_rgb_lidar.yml}"
STAGE2_TRAIN_CFG="${STAGE2_TRAIN_CFG:-${REPO_ROOT}/configs/baseline/occlusion_robustness/panoptic_seq1_runtime_mask_rgb_lidar/panoptic_seq1_panoptic_camhead_stage2_runtime_mask_rgb_lidar.yml}"
FINAL_EVAL_CFG="${FINAL_EVAL_CFG:-${REPO_ROOT}/configs/baseline/occlusion_robustness/panoptic_seq1_runtime_mask_rgb_lidar/panoptic_seq1_panoptic_final_eval_runtime_mask_rgb_lidar.yml}"
FINAL_EVAL_OCC_CFG="${FINAL_EVAL_OCC_CFG:-${REPO_ROOT}/configs/baseline/occlusion_robustness/panoptic_seq1_runtime_mask_rgb_lidar/panoptic_seq1_panoptic_final_eval_occluded_runtime_mask_rgb_lidar.yml}"
FINAL_EVAL_UNOCC_CFG="${FINAL_EVAL_UNOCC_CFG:-${REPO_ROOT}/configs/baseline/occlusion_robustness/panoptic_seq1_runtime_mask_rgb_lidar/panoptic_seq1_panoptic_final_eval_unoccluded_runtime_mask_rgb_lidar.yml}"

GPUS="${GPUS:-2}"
NUM_NODES="${NUM_NODES:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BATCH_SIZE_EVA="${BATCH_SIZE_EVA:-32}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PIN_MEMORY="${PIN_MEMORY:-1}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_OFFLINE="${WANDB_OFFLINE:-0}"

RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2="${RUN_STAGE2:-1}"
RUN_FINAL_EVAL="${RUN_FINAL_EVAL:-1}"
RUN_FINAL_EVAL_OCC="${RUN_FINAL_EVAL_OCC:-1}"
RUN_FINAL_EVAL_UNOCC="${RUN_FINAL_EVAL_UNOCC:-1}"

EXP_NAME_STAGE1="${EXP_NAME_STAGE1:-occlusion_robustness_seq1_runtime_mask_rgb_lidar}"
EXP_NAME_STAGE2="${EXP_NAME_STAGE2:-occlusion_robustness_seq1_runtime_mask_rgb_lidar_camhead}"
EXP_NAME_EVAL="${EXP_NAME_EVAL:-occlusion_robustness_seq1_runtime_mask_rgb_lidar_eval}"

RUN_STAMP="${RUN_STAMP:-$(date +'%Y%m%d_%H%M%S')}"
VERSION_STAGE1="${VERSION_STAGE1:-${RUN_STAMP}_stage1_train}"
VERSION_STAGE2="${VERSION_STAGE2:-${RUN_STAMP}_stage2_train}"
VERSION_EVAL_FULL="${VERSION_EVAL_FULL:-${RUN_STAMP}_final_eval}"
VERSION_EVAL_OCC="${VERSION_EVAL_OCC:-${RUN_STAMP}_final_eval_occluded}"
VERSION_EVAL_UNOCC="${VERSION_EVAL_UNOCC:-${RUN_STAMP}_final_eval_unoccluded}"

TMP_CFG_DIR="${TMP_CFG_DIR:-${REPO_ROOT}/logs/_generated_runtime_mask_cfgs/${RUN_STAMP}}"
mkdir -p "${TMP_CFG_DIR}"

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

EVAL_PER_FRAME_WORKERS="${EVAL_PER_FRAME_WORKERS:-${NUM_WORKERS}}"
EVAL_TARGET_MODALITY="${EVAL_TARGET_MODALITY:-lidar}"

log() {
  printf '[seq1-runtime-mask-rgb-lidar] %s\n' "$*"
}

find_best_ckpt() {
  local ckpt_dir="$1"
  local best_ckpt
  best_ckpt="$(find "${ckpt_dir}" -maxdepth 1 -type f -name '*.ckpt' ! -name 'last.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | awk '{print $2}')"
  if [[ -z "${best_ckpt}" && -f "${ckpt_dir}/last.ckpt" ]]; then
    best_ckpt="${ckpt_dir}/last.ckpt"
  fi
  if [[ -z "${best_ckpt}" || ! -f "${best_ckpt}" ]]; then
    return 1
  fi
  printf '%s\n' "${best_ckpt}"
}

render_cfg() {
  local src_cfg="$1"
  local dst_cfg="$2"
  local checkpoint_path="$3"
  local pretrained_camera_head_path="$4"
  uv run python - "${src_cfg}" "${dst_cfg}" "${checkpoint_path}" "${pretrained_camera_head_path}" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, ckpt, cam_ckpt = sys.argv[1:5]
with open(src, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

def normalize(value):
    if value in {"", "__KEEP__", "__NONE__"}:
        return value
    return str(Path(value).expanduser().resolve())

ckpt = normalize(ckpt)
cam_ckpt = normalize(cam_ckpt)

if ckpt != "__KEEP__":
    data["checkpoint_path"] = None if ckpt in {"", "__NONE__"} else ckpt
if cam_ckpt != "__KEEP__":
    data["pretrained_camera_head_path"] = None if cam_ckpt in {"", "__NONE__"} else cam_ckpt

dst_path = Path(dst)
dst_path.parent.mkdir(parents=True, exist_ok=True)
with dst_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, sort_keys=False)
PY
}

run_main() {
  local cfg="$1"
  shift
  uv run python "${REPO_ROOT}/main.py" -c "${cfg}" "${MAIN_ARGS[@]}" "$@"
}

find_test_predictions_pkl() {
  local run_dir="$1"
  local pred_pkl
  pred_pkl="$(find "${run_dir}" -maxdepth 1 -type f -name '*_test_predictions.pkl' | sort | head -n1)"
  if [[ -z "${pred_pkl}" || ! -f "${pred_pkl}" ]]; then
    return 1
  fi
  printf '%s\n' "${pred_pkl}"
}

run_per_frame_sensor_eval() {
  local pred_pkl="$1"
  local out_txt="$2"
  uv run python "${REPO_ROOT}/tools/eval_per_frame_sensor.py" \
    --pred-file "${pred_pkl}" \
    --pred-cameras-key pred_cameras_stream \
    --gt-cameras-key gt_cameras_stream \
    --target-modality "${EVAL_TARGET_MODALITY}" \
    --workers "${EVAL_PER_FRAME_WORKERS}" \
    | tee "${out_txt}"
}

STAGE1_BEST_CKPT="${STAGE1_BEST_CKPT:-}"
STAGE2_BEST_CKPT="${STAGE2_BEST_CKPT:-}"

if [[ "${RUN_STAGE1}" == "1" ]]; then
  log "Stage1 train config: ${STAGE1_TRAIN_CFG}"
  run_main "${STAGE1_TRAIN_CFG}" \
    --exp_name "${EXP_NAME_STAGE1}" \
    --version "${VERSION_STAGE1}"

  STAGE1_CKPT_DIR="${REPO_ROOT}/logs/${EXP_NAME_STAGE1}/${VERSION_STAGE1}"
  STAGE1_BEST_CKPT="$(find_best_ckpt "${STAGE1_CKPT_DIR}")" || {
    log "ERROR: no stage1 checkpoint found under ${STAGE1_CKPT_DIR}"
    exit 1
  }
else
  if [[ -z "${STAGE1_BEST_CKPT}" ]]; then
    log "ERROR: RUN_STAGE1=0 requires STAGE1_BEST_CKPT to be set."
    exit 1
  fi
  STAGE1_BEST_CKPT="$(realpath "${STAGE1_BEST_CKPT}")"
fi
log "Stage1 checkpoint: ${STAGE1_BEST_CKPT}"

if [[ "${RUN_STAGE2}" == "1" ]]; then
  STAGE2_CFG_RENDERED="${TMP_CFG_DIR}/stage2_train.yml"
  render_cfg "${STAGE2_TRAIN_CFG}" "${STAGE2_CFG_RENDERED}" "${STAGE1_BEST_CKPT}" "__KEEP__"
  log "Stage2 train config: ${STAGE2_CFG_RENDERED}"
  run_main "${STAGE2_CFG_RENDERED}" \
    --exp_name "${EXP_NAME_STAGE2}" \
    --version "${VERSION_STAGE2}"

  STAGE2_CKPT_DIR="${REPO_ROOT}/logs/${EXP_NAME_STAGE2}/${VERSION_STAGE2}"
  STAGE2_BEST_CKPT="$(find_best_ckpt "${STAGE2_CKPT_DIR}")" || {
    log "ERROR: no stage2 checkpoint found under ${STAGE2_CKPT_DIR}"
    exit 1
  }
else
  if [[ -z "${STAGE2_BEST_CKPT}" ]]; then
    log "ERROR: RUN_STAGE2=0 requires STAGE2_BEST_CKPT to be set."
    exit 1
  fi
  STAGE2_BEST_CKPT="$(realpath "${STAGE2_BEST_CKPT}")"
fi
log "Stage2 checkpoint: ${STAGE2_BEST_CKPT}"

run_eval() {
  local src_cfg="$1"
  local rendered_cfg="$2"
  local version="$3"
  local label="$4"
  render_cfg "${src_cfg}" "${rendered_cfg}" "${STAGE1_BEST_CKPT}" "${STAGE2_BEST_CKPT}"
  log "Final eval (${label}) config: ${rendered_cfg}"
  run_main "${rendered_cfg}" \
    --test \
    --exp_name "${EXP_NAME_EVAL}" \
    --version "${version}"
}

run_eval_per_frame_sensor() {
  local src_cfg="$1"
  local rendered_cfg="$2"
  local version="$3"
  local label="$4"
  local run_dir pred_pkl sensor_eval_txt
  render_cfg "${src_cfg}" "${rendered_cfg}" "${STAGE1_BEST_CKPT}" "${STAGE2_BEST_CKPT}"
  log "Final eval (${label}) config: ${rendered_cfg}"
  run_main "${rendered_cfg}" \
    --test \
    --save_test_preds \
    --exp_name "${EXP_NAME_EVAL}" \
    --version "${version}"

  run_dir="${REPO_ROOT}/logs/${EXP_NAME_EVAL}/${version}"
  pred_pkl="$(find_test_predictions_pkl "${run_dir}")" || {
    log "ERROR: no *_test_predictions.pkl found under ${run_dir}"
    exit 1
  }
  sensor_eval_txt="${run_dir}/per_frame_sensor_eval_${label}.txt"
  log "Per-frame sensor eval (${label}) predictions: ${pred_pkl}"
  run_per_frame_sensor_eval "${pred_pkl}" "${sensor_eval_txt}"
}

if [[ "${RUN_FINAL_EVAL}" == "1" ]]; then
  run_eval "${FINAL_EVAL_CFG}" "${TMP_CFG_DIR}/final_eval_full.yml" "${VERSION_EVAL_FULL}" "full"
fi

if [[ "${RUN_FINAL_EVAL_OCC}" == "1" ]]; then
  run_eval_per_frame_sensor "${FINAL_EVAL_OCC_CFG}" "${TMP_CFG_DIR}/final_eval_occluded.yml" "${VERSION_EVAL_OCC}" "occluded"
fi

if [[ "${RUN_FINAL_EVAL_UNOCC}" == "1" ]]; then
  run_eval_per_frame_sensor "${FINAL_EVAL_UNOCC_CFG}" "${TMP_CFG_DIR}/final_eval_unoccluded.yml" "${VERSION_EVAL_UNOCC}" "unoccluded"
fi

log "done"
