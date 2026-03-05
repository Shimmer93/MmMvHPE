#!/usr/bin/env python3
import argparse
from concurrent.futures import ProcessPoolExecutor
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.mpjpe import mpjpe_func, pampjpe_func
from misc.pose_enc import pose_encoding_to_extri_intri


_SEQ_RE = re.compile(r"(p\d+_a\d+)")
_FRAME_RE = re.compile(r"(\d+)$")
_KNOWN_SENSOR_MODALITY_PREFIXES = {
    "rgb",
    "depth",
    "lidar",
    "pc",
    "pointcloud",
    "mmwave",
    "radar",
    "ir",
    "thermal",
    "cam",
    "camera",
}

_EVAL_WORKER_CONTEXT: Optional[Dict[str, Any]] = None


def _iter_with_progress(iterable, *, desc: str, total: Optional[int], enabled: bool):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


def _to_numpy_maybe(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _set_eval_worker_context(context: Dict[str, Any]) -> None:
    global _EVAL_WORKER_CONTEXT
    _EVAL_WORKER_CONTEXT = context


def _init_eval_worker(context: Dict[str, Any]) -> None:
    _set_eval_worker_context(context)


def _parse_sensor_index_map(text: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    raw = str(text).strip()
    if not raw:
        return out
    for item in raw.split(","):
        part = item.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                f"Invalid --sensor-index-by-modality entry `{part}`. Expected `modality:index`."
            )
        mod, idx = part.split(":", 1)
        mod = mod.strip().lower()
        idx = idx.strip()
        if not mod:
            raise ValueError(f"Invalid modality in --sensor-index-by-modality entry `{part}`.")
        out[mod] = int(idx)
    return out


def _resolve_camera_key(data: dict, requested_key: str, fallback_keys: Sequence[str]) -> str:
    if requested_key in data:
        return requested_key
    if requested_key not in fallback_keys:
        raise ValueError(f"Requested camera key `{requested_key}` is not in prediction file keys.")
    for key in fallback_keys:
        if key in data:
            return key
    raise ValueError(
        f"None of the camera keys exist in prediction file: requested={requested_key}, "
        f"fallbacks={list(fallback_keys)}."
    )


def _seq_base_name_from_sample_id(sample_id: Optional[str]) -> str:
    if sample_id is None:
        return "__all__"
    s = str(sample_id)
    m = _SEQ_RE.search(s)
    if m is not None:
        return m.group(1)
    parts = s.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return s


def _sensor_combo_from_sample_id(sample_id: Optional[str]) -> Optional[str]:
    if sample_id is None:
        return None
    parts = str(sample_id).split("_")
    if len(parts) < 3:
        return None
    tokens = []
    seen = set()
    for i in range(len(parts) - 2):
        mod = parts[i].strip().lower()
        family = parts[i + 1].strip().lower()
        idx = parts[i + 2].strip()
        if mod not in _KNOWN_SENSOR_MODALITY_PREFIXES:
            continue
        if (not family) or (not idx.isdigit()):
            continue
        token = f"{mod}_{family}_{idx}"
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    if len(tokens) == 0:
        return None
    return "+".join(tokens)


def _get_stream_metadata_for_sample(
    data,
    sample_idx: int,
) -> Tuple[Optional[Sequence[str]], Optional[Sequence[int]]]:
    global_modalities = data.get("camera_stream_modalities", None)
    global_sensor_indices = data.get("camera_stream_sensor_indices", None)
    if global_modalities is not None and global_sensor_indices is not None:
        return global_modalities, global_sensor_indices

    per_sample_modalities = data.get("camera_stream_modalities_per_sample", None)
    per_sample_sensor_indices = data.get("camera_stream_sensor_indices_per_sample", None)
    if per_sample_modalities is None or per_sample_sensor_indices is None:
        return None, None
    if sample_idx >= len(per_sample_modalities) or sample_idx >= len(per_sample_sensor_indices):
        return None, None
    return per_sample_modalities[sample_idx], per_sample_sensor_indices[sample_idx]


def _sensor_combo_from_stream_metadata_for_sample(data, sample_idx: int) -> Optional[str]:
    stream_modalities, stream_sensor_indices = _get_stream_metadata_for_sample(data, sample_idx)
    if stream_modalities is None or stream_sensor_indices is None:
        return None
    if len(stream_modalities) != len(stream_sensor_indices):
        return None
    if len(stream_modalities) == 0:
        return None

    tokens = []
    seen = set()
    for m, idx in zip(stream_modalities, stream_sensor_indices):
        mod = str(m).strip().lower()
        if not mod:
            continue
        token = f"{mod}_{int(idx)}"
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    if len(tokens) == 0:
        return None
    return "+".join(tokens)


def _seq_group_name_for_sample(data, sample_ids: Sequence[str], sample_idx: int) -> str:
    sid = sample_ids[sample_idx] if sample_idx < len(sample_ids) else None
    base = _seq_base_name_from_sample_id(sid)
    combo_stream = _sensor_combo_from_stream_metadata_for_sample(data, sample_idx)
    combo_sample_id = _sensor_combo_from_sample_id(sid)
    if combo_stream is None and combo_sample_id is None:
        return base
    if combo_stream is not None and combo_sample_id is not None:
        if combo_stream == combo_sample_id:
            combo = combo_stream
        else:
            combo = f"stream:{combo_stream}__id:{combo_sample_id}"
    elif combo_stream is not None:
        combo = f"stream:{combo_stream}"
    else:
        combo = f"id:{combo_sample_id}"
    return f"{base}__{combo}"


def _frame_index_from_sample_id(sample_id: Optional[str]) -> Optional[int]:
    if sample_id is None:
        return None
    m = _FRAME_RE.search(str(sample_id))
    if m is None:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _build_seq_group_names(data, sample_ids: Sequence[str]):
    return [_seq_group_name_for_sample(data, sample_ids, i) for i in range(len(sample_ids))]


def _get_modalities_for_sample(data, sample_idx: int) -> Optional[Sequence[str]]:
    global_modalities = data.get("camera_modalities", None)
    if global_modalities is not None:
        return global_modalities
    per_sample = data.get("camera_modalities_per_sample", None)
    if per_sample is None or sample_idx >= len(per_sample):
        return None
    return per_sample[sample_idx]


def _get_modality_index(
    data,
    sample_idx: int,
    modality: str,
    fallback_idx: Optional[int],
) -> Optional[int]:
    modalities = _get_modalities_for_sample(data, sample_idx)
    if modalities is not None:
        target = str(modality).lower()
        for i, m in enumerate(modalities):
            if str(m).lower() == target:
                return i
    return fallback_idx


def _is_stream_camera_key(camera_key: str) -> bool:
    return str(camera_key).endswith("_stream")


def _get_camera_index(
    data,
    sample_idx: int,
    modality: str,
    fallback_idx: Optional[int],
    use_stream_index: bool,
    sensor_idx: Optional[int],
) -> Optional[int]:
    if not use_stream_index:
        return _get_modality_index(data, sample_idx, modality, fallback_idx)

    stream_modalities, stream_sensor_indices = _get_stream_metadata_for_sample(data, sample_idx)
    if stream_modalities is None or stream_sensor_indices is None:
        if sensor_idx not in (None, 0):
            raise ValueError(
                f"Requested sensor_idx={sensor_idx} for modality `{modality}` but stream metadata is missing."
            )
        return _get_modality_index(data, sample_idx, modality, fallback_idx)

    if len(stream_modalities) != len(stream_sensor_indices):
        raise ValueError(
            f"Invalid stream metadata lengths: modalities={len(stream_modalities)}, "
            f"sensor_indices={len(stream_sensor_indices)}."
        )

    target_modality = str(modality).lower()
    candidates = [i for i, m in enumerate(stream_modalities) if str(m).lower() == target_modality]
    if len(candidates) == 0:
        return None
    if sensor_idx is None:
        return int(candidates[0])

    matches = []
    for i in candidates:
        if int(stream_sensor_indices[i]) == int(sensor_idx):
            matches.append(i)
    if len(matches) == 0:
        return None
    return int(matches[0])


def _get_sample_keypoints(arr, sample_idx: int) -> Optional[np.ndarray]:
    if arr is None:
        return None
    if isinstance(arr, np.ndarray) and arr.dtype != object:
        if arr.ndim == 3:
            return arr[sample_idx]
        if arr.ndim == 2:
            return arr
        return None

    if sample_idx >= len(arr):
        return None
    item = arr[sample_idx]
    if item is None:
        return None
    item = _to_numpy_maybe(item)
    if item is None:
        return None
    if item.ndim == 3:
        return item[-1]
    return item


def _extract_camera_encoding(arr, sample_idx: int, modality_idx: Optional[int]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    if isinstance(arr, np.ndarray) and arr.dtype != object:
        if arr.ndim == 3:
            if modality_idx is None:
                modality_idx = 0
            return arr[sample_idx, modality_idx]
        if arr.ndim == 2:
            return arr[sample_idx]
        return None

    if sample_idx >= len(arr):
        return None
    item = arr[sample_idx]
    if item is None:
        return None
    item = _to_numpy_maybe(item)
    if item is None:
        return None

    if item.ndim == 1 and item.shape[0] == 9:
        return item
    if item.ndim == 2 and item.shape[-1] == 9:
        if modality_idx is None:
            modality_idx = 0
        if modality_idx >= item.shape[0]:
            return None
        return item[modality_idx]
    if item.ndim == 3 and item.shape[-1] == 9:
        if modality_idx is None:
            modality_idx = 0
        if modality_idx >= item.shape[0]:
            return None
        return item[modality_idx, -1]
    return None


def _get_sample_lidar_center(data, sample_idx: int) -> Optional[np.ndarray]:
    centers = data.get("input_lidar_center", None)
    if centers is None:
        return None

    if isinstance(centers, np.ndarray) and centers.dtype != object:
        if centers.ndim == 2 and centers.shape[-1] == 3:
            if sample_idx >= centers.shape[0]:
                return None
            center = centers[sample_idx]
        elif centers.ndim == 1 and centers.shape[0] == 3:
            center = centers
        else:
            return None
    else:
        if sample_idx >= len(centers):
            return None
        center = centers[sample_idx]
        if center is None:
            return None
        center = _to_numpy_maybe(center)
        if center is None:
            return None

    center = np.asarray(center, dtype=np.float32).reshape(-1)
    if center.shape[0] != 3 or (not np.isfinite(center).all()):
        return None
    return center


def _inverse_lidar_camera_center(
    cam_encoding: Optional[np.ndarray],
    lidar_center: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if cam_encoding is None or lidar_center is None:
        return cam_encoding
    cam = np.asarray(cam_encoding, dtype=np.float32).reshape(-1)
    if cam.shape[0] < 3:
        return cam_encoding
    out = cam.copy()
    out[:3] = out[:3] + np.asarray(lidar_center, dtype=np.float32).reshape(3)
    return out


def _pose_encoding_to_extrinsic(pose_encoding: np.ndarray, pose_encoding_type: str) -> np.ndarray:
    if pose_encoding_type == "absT_quaR_FoV":
        pe = np.asarray(pose_encoding, dtype=np.float32).reshape(-1)
        if pe.shape[0] < 7:
            raise ValueError(f"Invalid pose encoding shape for {pose_encoding_type}: {pe.shape}")
        t = pe[:3]
        q = pe[3:7]
        qn = float(np.linalg.norm(q))
        if not np.isfinite(qn) or qn < 1e-12:
            raise ValueError("Invalid quaternion in pose encoding.")
        x, y, z, w = (q / qn).tolist()  # XYZW
        r = np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )
        return np.concatenate([r, t.reshape(3, 1).astype(np.float32)], axis=1)

    pe = torch.as_tensor(pose_encoding, dtype=torch.float32).view(1, 1, -1)
    extrinsics, _ = pose_encoding_to_extri_intri(
        pe,
        image_size_hw=None,
        pose_encoding_type=pose_encoding_type,
        build_intrinsics=False,
    )
    return extrinsics[0, 0].detach().cpu().numpy().astype(np.float32)


def _transform_points(points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    r = extrinsic[:, :3]
    t = extrinsic[:, 3]
    return points @ r.T + t[None, :]


def _format_mm(x_meters: float) -> str:
    return f"{x_meters:.6f} m ({x_meters * 1000.0:.3f} mm)"


def _extract_valid_keypoint_pair(pred_all, gt_all, sample_idx: int):
    pred_kps = _get_sample_keypoints(pred_all, sample_idx)
    gt_kps = _get_sample_keypoints(gt_all, sample_idx)
    if pred_kps is None or gt_kps is None:
        return None, None
    pred_kps = _to_numpy_maybe(pred_kps)
    gt_kps = _to_numpy_maybe(gt_kps)
    if pred_kps is None or gt_kps is None:
        return None, None
    if pred_kps.shape != gt_kps.shape:
        return None, None
    if pred_kps.ndim != 2 or pred_kps.shape[-1] != 3:
        return None, None
    if not np.isfinite(pred_kps).all() or not np.isfinite(gt_kps).all():
        return None, None
    return pred_kps.astype(np.float32), gt_kps.astype(np.float32)


def _decode_camera_extrinsic_for_sample(
    data: dict,
    camera_key: str,
    sample_idx: int,
    modality: str,
    fallback_idx: Optional[int],
    sensor_idx: Optional[int],
    pose_encoding_type: str,
) -> Optional[np.ndarray]:
    cameras = data.get(camera_key, None)
    if cameras is None:
        return None
    use_stream_index = _is_stream_camera_key(camera_key)
    camera_idx = _get_camera_index(
        data=data,
        sample_idx=sample_idx,
        modality=modality,
        fallback_idx=fallback_idx,
        use_stream_index=use_stream_index,
        sensor_idx=sensor_idx,
    )
    cam = _extract_camera_encoding(cameras, sample_idx, camera_idx)
    if str(modality).lower() == "lidar":
        center = _get_sample_lidar_center(data, sample_idx)
        cam = _inverse_lidar_camera_center(cam, center)
    if cam is None:
        return None
    cam = np.asarray(cam, dtype=np.float32).reshape(-1)
    if cam.shape[0] < 9 or (not np.isfinite(cam).all()):
        return None
    try:
        extr = _pose_encoding_to_extrinsic(cam, pose_encoding_type)
    except Exception:
        return None
    if not np.isfinite(extr).all():
        return None
    return extr.astype(np.float32)


def _smooth_temporal_by_sequence(
    points: np.ndarray,
    seq_names: Sequence[str],
    frame_indices: Sequence[Optional[int]],
    window: int,
    mode: str = "centered",
) -> np.ndarray:
    if window <= 1:
        return points
    if mode not in {"centered", "causal"}:
        raise ValueError(f"Unknown smoothing mode `{mode}`. Expected one of {{'centered','causal'}}.")
    if mode == "centered" and window % 2 == 0:
        raise ValueError(f"Smoothing window must be odd for centered mode, got {window}.")

    points = np.asarray(points, dtype=np.float32)
    out = points.copy()
    half = window // 2
    seq_to_indices = {}
    for i, seq in enumerate(seq_names):
        seq_to_indices.setdefault(seq, []).append(i)

    for seq, idxs in seq_to_indices.items():
        if len(idxs) <= 1:
            continue
        if all(frame_indices[i] is not None for i in idxs):
            idxs_sorted = sorted(idxs, key=lambda i: int(frame_indices[i]))
        else:
            idxs_sorted = list(idxs)

        for p, i_center in enumerate(idxs_sorted):
            if mode == "centered":
                l = max(0, p - half)
                r = min(len(idxs_sorted), p + half + 1)
            else:
                l = max(0, p - window + 1)
                r = p + 1
            out[i_center] = points[idxs_sorted[l:r]].mean(axis=0)
    return out


def _process_sample_worker(sample_idx: int):
    if _EVAL_WORKER_CONTEXT is None:
        raise RuntimeError("Evaluation worker context is not initialized.")
    ctx = _EVAL_WORKER_CONTEXT

    pred_kps, gt_kps = _extract_valid_keypoint_pair(ctx["pred_all"], ctx["gt_all"], sample_idx)
    if pred_kps is None or gt_kps is None:
        return ("skip_kps", sample_idx, None, None, None, None)

    pred_extr = _decode_camera_extrinsic_for_sample(
        data=ctx["data"],
        camera_key=ctx["pred_camera_key"],
        sample_idx=sample_idx,
        modality=ctx["target_modality"],
        fallback_idx=ctx["fallback_idx"],
        sensor_idx=ctx["target_sensor_idx"],
        pose_encoding_type=ctx["pose_encoding_type"],
    )
    gt_extr = _decode_camera_extrinsic_for_sample(
        data=ctx["data"],
        camera_key=ctx["gt_camera_key"],
        sample_idx=sample_idx,
        modality=ctx["target_modality"],
        fallback_idx=ctx["fallback_idx"],
        sensor_idx=ctx["target_sensor_idx"],
        pose_encoding_type=ctx["pose_encoding_type"],
    )
    if pred_extr is None or gt_extr is None:
        return ("skip_cam", sample_idx, None, None, None, None)

    sid = ctx["sample_ids"][sample_idx] if sample_idx < len(ctx["sample_ids"]) else None
    seq_name = (
        str(ctx["seq_group_names"][sample_idx])
        if sample_idx < len(ctx["seq_group_names"])
        else _seq_base_name_from_sample_id(sid)
    )
    frame_idx = _frame_index_from_sample_id(sid)
    pred_proj = _transform_points(pred_kps, pred_extr).astype(np.float32)
    gt_proj = _transform_points(gt_kps, gt_extr).astype(np.float32)
    return ("ok", sample_idx, seq_name, frame_idx, pred_proj, gt_proj)


def _run_sample_worker_pool(
    context: Dict[str, Any],
    num_samples: int,
    workers: int,
    show_progress: bool,
    desc: str = "project per-frame",
):
    if workers <= 1:
        _set_eval_worker_context(context)
        iterator = (_process_sample_worker(i) for i in range(num_samples))
        for item in _iter_with_progress(iterator, desc=desc, total=num_samples, enabled=show_progress):
            yield item
        return

    chunksize = max(1, min(256, num_samples // max(workers * 4, 1)))
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_eval_worker,
        initargs=(context,),
    ) as executor:
        iterator = executor.map(_process_sample_worker, range(num_samples), chunksize=chunksize)
        for item in _iter_with_progress(iterator, desc=desc, total=num_samples, enabled=show_progress):
            yield item


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate per-frame sensor-space MPJPE/PA-MPJPE by directly applying framewise "
            "canonical-to-target extrinsics (no sequence-fixed anchor)."
        )
    )
    parser.add_argument("--pred-file", type=str, required=True, help="Path to *_test_predictions.pkl")
    parser.add_argument("--pred-keypoints-key", type=str, default="pred_keypoints")
    parser.add_argument("--gt-keypoints-key", type=str, default="gt_keypoints")
    parser.add_argument("--pred-cameras-key", type=str, default="pred_cameras_stream")
    parser.add_argument("--gt-cameras-key", type=str, default="gt_cameras_stream")
    parser.add_argument("--pose-encoding-type", type=str, default="absT_quaR_FoV")
    parser.add_argument("--target-modality", type=str, default="lidar")
    parser.add_argument(
        "--sensor-index-by-modality",
        type=str,
        default="",
        help="Comma-separated `modality:sensor_idx` (e.g., `lidar:0,rgb:1`) for stream camera keys.",
    )
    parser.add_argument(
        "--lidar-modality-index",
        type=int,
        default=None,
        help="Fallback LiDAR modality index if camera modality names are unavailable.",
    )
    parser.add_argument(
        "--rgb-modality-index",
        type=int,
        default=None,
        help="Fallback RGB modality index if camera modality names are unavailable.",
    )
    parser.add_argument("--pelvis-index", type=int, default=0)
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Temporal smoothing window per sequence. For centered mode it must be odd. 1 disables smoothing.",
    )
    parser.add_argument(
        "--smooth-on",
        type=str,
        default="pred",
        choices=["pred", "gt", "both"],
    )
    parser.add_argument(
        "--smooth-mode",
        type=str,
        default="centered",
        choices=["centered", "causal"],
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for per-sample projection. 1 disables multiprocessing.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="If >0, evaluate only the first N samples.")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    show_progress = not args.no_progress
    if args.workers <= 0:
        raise ValueError(f"--workers must be >= 1, got {args.workers}")
    with open(args.pred_file, "rb") as f:
        data = pickle.load(f)

    pred_camera_key = _resolve_camera_key(
        data=data,
        requested_key=args.pred_cameras_key,
        fallback_keys=["pred_cameras_stream", "pred_cameras"],
    )
    gt_camera_key = _resolve_camera_key(
        data=data,
        requested_key=args.gt_cameras_key,
        fallback_keys=["gt_cameras_stream", "gt_cameras"],
    )

    pred_all = data.get(args.pred_keypoints_key, None)
    gt_all = data.get(args.gt_keypoints_key, None)
    if pred_all is None or gt_all is None:
        raise ValueError(
            f"Missing keypoints in file. Found pred={pred_all is not None}, gt={gt_all is not None}."
        )

    sample_ids = data.get("sample_ids", None)
    if sample_ids is None:
        sample_ids = [f"sample_{i}" for i in range(len(pred_all))]
    num_samples = len(sample_ids)
    if args.max_samples > 0:
        num_samples = min(num_samples, int(args.max_samples))
        sample_ids = sample_ids[:num_samples]

    target_modality = str(args.target_modality).lower()
    sensor_index_by_modality = _parse_sensor_index_map(args.sensor_index_by_modality)
    if target_modality not in sensor_index_by_modality:
        sensor_index_by_modality[target_modality] = 0
    modality_fallback_indices: Dict[str, Optional[int]] = {
        "lidar": args.lidar_modality_index,
        "rgb": args.rgb_modality_index,
    }
    fallback_idx = modality_fallback_indices.get(target_modality, None)
    target_sensor_idx = sensor_index_by_modality.get(target_modality, 0)
    seq_group_names = _build_seq_group_names(data, sample_ids)

    pred_proj_all = []
    gt_proj_all = []
    kept_seq_names = []
    kept_frame_indices = []
    drop_missing_kps = 0
    drop_missing_cam = 0

    worker_context = {
        "data": data,
        "pred_all": pred_all,
        "gt_all": gt_all,
        "sample_ids": sample_ids,
        "seq_group_names": seq_group_names,
        "pred_camera_key": pred_camera_key,
        "gt_camera_key": gt_camera_key,
        "target_modality": target_modality,
        "fallback_idx": fallback_idx,
        "target_sensor_idx": target_sensor_idx,
        "pose_encoding_type": args.pose_encoding_type,
    }
    for status, _i, seq_name, frame_idx, pred_proj, gt_proj in _run_sample_worker_pool(
        context=worker_context,
        num_samples=num_samples,
        workers=int(args.workers),
        show_progress=show_progress,
        desc="project per-frame",
    ):
        if status == "skip_kps":
            drop_missing_kps += 1
            continue
        if status == "skip_cam":
            drop_missing_cam += 1
            continue
        if status != "ok":
            continue
        pred_proj_all.append(pred_proj)
        gt_proj_all.append(gt_proj)
        kept_seq_names.append(seq_name)
        kept_frame_indices.append(frame_idx)

    if len(pred_proj_all) == 0:
        raise RuntimeError("No valid samples found for evaluation.")

    pred_new = np.stack(pred_proj_all, axis=0)
    gt_new = np.stack(gt_proj_all, axis=0)

    if args.smooth_window > 1:
        if args.smooth_on in {"pred", "both"}:
            pred_new = _smooth_temporal_by_sequence(
                pred_new,
                seq_names=kept_seq_names,
                frame_indices=kept_frame_indices,
                window=args.smooth_window,
                mode=args.smooth_mode,
            )
        if args.smooth_on in {"gt", "both"}:
            gt_new = _smooth_temporal_by_sequence(
                gt_new,
                seq_names=kept_seq_names,
                frame_indices=kept_frame_indices,
                window=args.smooth_window,
                mode=args.smooth_mode,
            )

    mpjpe_new = float(mpjpe_func(pred_new, gt_new, reduce=True))
    pampjpe_new = float(pampjpe_func(pred_new, gt_new, reduce=True))

    pelvis = int(args.pelvis_index)
    pred_centered = pred_new - pred_new[:, pelvis:pelvis + 1, :]
    gt_centered = gt_new - gt_new[:, pelvis:pelvis + 1, :]
    mpjpe_centered = float(mpjpe_func(pred_centered, gt_centered, reduce=True))

    print(f"[INFO] Loaded: {args.pred_file}")
    print("[INFO] Projection mode: per_frame_sensor (framewise canonical->target, no sequence anchor)")
    print(f"[INFO] Camera keys: pred={pred_camera_key}, gt={gt_camera_key}")
    print(f"[INFO] Worker processes: {int(args.workers)}")
    print(f"[INFO] Target modality: {target_modality}, sensor_idx={target_sensor_idx}")
    print(f"[INFO] Smoothing: window={args.smooth_window}, target={args.smooth_on}, mode={args.smooth_mode}")
    print(f"[INFO] Valid samples: {len(pred_proj_all)}/{num_samples}")
    if drop_missing_kps > 0:
        print(f"[INFO] Dropped samples (invalid keypoints): {drop_missing_kps}")
    if drop_missing_cam > 0:
        print(f"[INFO] Dropped samples (missing/invalid cameras): {drop_missing_cam}")
    print(f"[RESULT] MPJPE (sensor frame): {_format_mm(mpjpe_new)}")
    print(f"[RESULT] PA-MPJPE (sensor frame): {_format_mm(pampjpe_new)}")
    print(f"[RESULT] MPJPE (sensor frame, pelvis-centered @ joint {pelvis}): {_format_mm(mpjpe_centered)}")


if __name__ == "__main__":
    main()
