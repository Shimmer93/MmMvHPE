from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from metrics.mpjpe import mpjpe_func, pampjpe_func, pcmpjpe_func


@dataclass(frozen=True)
class FrameMetricResult:
    sample_index: int
    frame_id: int
    valid: bool
    reason: str | None
    mpjpe: float
    pa_mpjpe: float
    pc_mpjpe: float


def invalid_frame_result(sample_index: int, frame_id: int, reason: str) -> FrameMetricResult:
    return FrameMetricResult(
        sample_index=sample_index,
        frame_id=frame_id,
        valid=False,
        reason=reason,
        mpjpe=float("nan"),
        pa_mpjpe=float("nan"),
        pc_mpjpe=float("nan"),
    )


def evaluate_frame_metrics(
    pred_keypoints: np.ndarray,
    gt_keypoints: np.ndarray,
    *,
    sample_index: int,
    frame_id: int,
    pelvis_idx: int = 2,
) -> FrameMetricResult:
    pred = np.asarray(pred_keypoints, dtype=np.float32)
    gt = np.asarray(gt_keypoints, dtype=np.float32)
    if pred.shape != (19, 3):
        raise ValueError(f"Predicted Panoptic joints must have shape (19,3), got {pred.shape}.")
    if gt.shape != (19, 3):
        raise ValueError(f"GT Panoptic joints must have shape (19,3), got {gt.shape}.")
    if not np.isfinite(pred).all():
        return invalid_frame_result(sample_index, frame_id, "non_finite_prediction")
    if not np.isfinite(gt).all():
        return invalid_frame_result(sample_index, frame_id, "non_finite_ground_truth")

    pred_batch = pred[None, ...]
    gt_batch = gt[None, ...]
    return FrameMetricResult(
        sample_index=sample_index,
        frame_id=frame_id,
        valid=True,
        reason=None,
        mpjpe=float(mpjpe_func(pred_batch, gt_batch, reduce=True)),
        pa_mpjpe=float(pampjpe_func(pred_batch, gt_batch, reduce=True)),
        pc_mpjpe=float(pcmpjpe_func(pred_batch, gt_batch, pelvis_idx=pelvis_idx, reduce=True)),
    )


def aggregate_segment_metrics(
    *,
    sequence_name: str,
    camera_name: str,
    segment_index: int,
    segment_length: int,
    start_frame_id: int,
    end_frame_id: int,
    sample_indices: list[int] | tuple[int, ...],
    frame_results: list[FrameMetricResult],
) -> dict:
    if not frame_results:
        raise ValueError("aggregate_segment_metrics requires at least one frame result.")

    valid_results = [item for item in frame_results if item.valid]
    invalid_counts = Counter(item.reason for item in frame_results if not item.valid and item.reason is not None)

    def _metric_values(name: str) -> list[float]:
        return [float(getattr(item, name)) for item in valid_results]

    def _safe_mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    def _safe_max(values: list[float]) -> float:
        return float(np.max(values)) if values else float("nan")

    mpjpe_values = _metric_values("mpjpe")
    pa_values = _metric_values("pa_mpjpe")
    pc_values = _metric_values("pc_mpjpe")

    return {
        "sequence_name": sequence_name,
        "camera_name": camera_name,
        "segment_index": int(segment_index),
        "segment_length": int(segment_length),
        "start_frame_id": int(start_frame_id),
        "end_frame_id": int(end_frame_id),
        "num_frames": int(len(frame_results)),
        "sample_indices": list(sample_indices),
        "frame_ids": [int(item.frame_id) for item in frame_results],
        "num_valid_frames": int(len(valid_results)),
        "num_invalid_frames": int(len(frame_results) - len(valid_results)),
        "invalid_reasons": dict(sorted(invalid_counts.items())),
        "mpjpe_mean": _safe_mean(mpjpe_values),
        "pa_mpjpe_mean": _safe_mean(pa_values),
        "pc_mpjpe_mean": _safe_mean(pc_values),
        "mpjpe_max": _safe_max(mpjpe_values),
        "pa_mpjpe_max": _safe_max(pa_values),
        "pc_mpjpe_max": _safe_max(pc_values),
        "frame_metrics": [
            {
                "sample_index": int(item.sample_index),
                "frame_id": int(item.frame_id),
                "valid": bool(item.valid),
                "reason": item.reason,
                "mpjpe": float(item.mpjpe),
                "pa_mpjpe": float(item.pa_mpjpe),
                "pc_mpjpe": float(item.pc_mpjpe),
            }
            for item in frame_results
        ],
    }
