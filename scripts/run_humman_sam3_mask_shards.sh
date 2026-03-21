#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/opt/data/humman_cropped}"
MASK_ROOT="${MASK_ROOT:-${DATA_ROOT}/sam_segmentation_mask}"
SEGMENTOR_PATH="${SEGMENTOR_PATH:-/opt/data/SAM3_checkpoint}"
NUM_SHARDS="${NUM_SHARDS:-2}"
DEVICE_IDS_RAW="${DEVICE_IDS:-0}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/humman_sam3_masks}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

IFS=',' read -r -a DEVICE_IDS <<< "${DEVICE_IDS_RAW}"
if [[ "${#DEVICE_IDS[@]}" -lt 1 ]]; then
  echo "[humman-sam3-mask-launcher] ERROR: DEVICE_IDS resolved to an empty list" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

echo "[humman-sam3-mask-launcher] data_root=${DATA_ROOT}"
echo "[humman-sam3-mask-launcher] mask_root=${MASK_ROOT}"
echo "[humman-sam3-mask-launcher] segmentor_path=${SEGMENTOR_PATH}"
echo "[humman-sam3-mask-launcher] num_shards=${NUM_SHARDS}"
echo "[humman-sam3-mask-launcher] device_ids=${DEVICE_IDS_RAW}"
echo "[humman-sam3-mask-launcher] log_dir=${LOG_DIR}"

pids=()
for (( shard=0; shard<NUM_SHARDS; shard++ )); do
  device="${DEVICE_IDS[$((shard % ${#DEVICE_IDS[@]}))]}"
  log_file="${LOG_DIR}/shard_${shard}_of_${NUM_SHARDS}_gpu${device}.log"
  summary_file="${LOG_DIR}/shard_${shard}_of_${NUM_SHARDS}_gpu${device}.json"
  echo "[humman-sam3-mask-launcher] launching shard=${shard} gpu=${device}"
  CUDA_VISIBLE_DEVICES="${device}" \
    uv run python "${REPO_ROOT}/tools/generate_humman_sam3_segmentation_masks.py" \
      --data-root "${DATA_ROOT}" \
      --mask-root "${MASK_ROOT}" \
      --segmentor-path "${SEGMENTOR_PATH}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard}" \
      --summary-json "${summary_file}" \
      ${EXTRA_ARGS} \
      > "${log_file}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"
