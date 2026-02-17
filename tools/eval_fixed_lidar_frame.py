#!/usr/bin/env python3
import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

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


_SEQ_RE = re.compile(r"(p\d+_a\d+)")
_FRAME_RE = re.compile(r"(\d+)$")


def _seq_name_from_sample_id(sample_id: Optional[str]) -> str:
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


def _frame_index_from_sample_id(sample_id: Optional[str]) -> Optional[int]:
    if sample_id is None:
        return None
    s = str(sample_id)
    m = _FRAME_RE.search(s)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _get_modalities_for_sample(data, sample_idx: int) -> Optional[Sequence[str]]:
    global_modalities = data.get("camera_modalities", None)
    if global_modalities is not None:
        return global_modalities
    per_sample = data.get("camera_modalities_per_sample", None)
    if per_sample is None:
        return None
    if sample_idx >= len(per_sample):
        return None
    return per_sample[sample_idx]


def _get_lidar_modality_index(data, sample_idx: int, fallback_idx: Optional[int]) -> Optional[int]:
    modalities = _get_modalities_for_sample(data, sample_idx)
    if modalities is not None:
        for i, m in enumerate(modalities):
            if str(m).lower() == "lidar":
                return i
    return fallback_idx


def _get_sample_keypoints(arr, sample_idx: int) -> Optional[np.ndarray]:
    if arr is None:
        return None
    if isinstance(arr, np.ndarray) and arr.dtype != object:
        if arr.ndim == 3:
            return arr[sample_idx]
        if arr.ndim == 2:
            return arr
        return None
    item = arr[sample_idx]
    if item is None:
        return None
    item = _to_numpy_maybe(item)
    if item is None:
        return None
    if item.ndim == 3:
        # [T, J, C] -> use last frame.
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
        # [M, S, 9] -> use last frame.
        if modality_idx is None:
            modality_idx = 0
        if modality_idx >= item.shape[0]:
            return None
        return item[modality_idx, -1]
    return None


def _find_first_valid_lidar_camera(
    data,
    camera_key: str,
    num_samples: int,
    fallback_lidar_idx: Optional[int],
    show_progress: bool = True,
) -> Tuple[np.ndarray, int, int]:
    cameras = data.get(camera_key, None)
    if cameras is None:
        raise ValueError(f"Missing `{camera_key}` in prediction file.")

    for i in _iter_with_progress(
        range(num_samples),
        desc=f"search {camera_key}",
        total=num_samples,
        enabled=show_progress,
    ):
        lidar_idx = _get_lidar_modality_index(data, i, fallback_lidar_idx)
        cam = _extract_camera_encoding(cameras, i, lidar_idx)
        if cam is None:
            continue
        if np.isfinite(cam).all():
            return cam.astype(np.float32), i, int(0 if lidar_idx is None else lidar_idx)
    raise ValueError(f"Could not find a finite LiDAR camera in `{camera_key}`.")


def _build_sequence_reference_cameras(
    data,
    camera_key: str,
    sample_ids: Sequence[str],
    num_samples: int,
    fallback_lidar_idx: Optional[int],
    show_progress: bool = True,
):
    cameras = data.get(camera_key, None)
    if cameras is None:
        raise ValueError(f"Missing `{camera_key}` in prediction file.")

    seq_to_camera = {}
    seq_to_meta = {}
    iterator = _iter_with_progress(
        range(num_samples),
        desc=f"build seq refs {camera_key}",
        total=num_samples,
        enabled=show_progress,
    )
    for i in iterator:
        sid = sample_ids[i] if i < len(sample_ids) else None
        seq_name = _seq_name_from_sample_id(sid)
        if seq_name in seq_to_camera:
            continue
        lidar_idx = _get_lidar_modality_index(data, i, fallback_lidar_idx)
        cam = _extract_camera_encoding(cameras, i, lidar_idx)
        if cam is None or not np.isfinite(cam).all():
            continue
        seq_to_camera[seq_name] = cam.astype(np.float32)
        seq_to_meta[seq_name] = (i, int(0 if lidar_idx is None else lidar_idx))

    if len(seq_to_camera) == 0:
        raise ValueError(f"Could not find any finite LiDAR camera in `{camera_key}`.")
    return seq_to_camera, seq_to_meta


def _pose_encoding_to_extrinsic(pose_encoding: np.ndarray, pose_encoding_type: str) -> np.ndarray:
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


def _smooth_temporal_by_sequence(
    points: np.ndarray,
    seq_names: Sequence[str],
    frame_indices: Sequence[Optional[int]],
    window: int,
) -> np.ndarray:
    if window <= 1:
        return points
    if window % 2 == 0:
        raise ValueError(f"Smoothing window must be odd, got {window}.")

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
            l = max(0, p - half)
            r = min(len(idxs_sorted), p + half + 1)
            nbr = idxs_sorted[l:r]
            out[i_center] = points[nbr].mean(axis=0)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-sensor LiDAR-frame MPJPE/PA-MPJPE from saved test predictions."
    )
    parser.add_argument("--pred-file", type=str, required=True, help="Path to *_test_predictions.pkl")
    parser.add_argument("--pred-keypoints-key", type=str, default="pred_keypoints")
    parser.add_argument("--gt-keypoints-key", type=str, default="gt_keypoints")
    parser.add_argument("--pred-cameras-key", type=str, default="pred_cameras")
    parser.add_argument("--gt-cameras-key", type=str, default="gt_cameras")
    parser.add_argument("--pose-encoding-type", type=str, default="absT_quaR_FoV")
    parser.add_argument(
        "--lidar-modality-index",
        type=int,
        default=None,
        help="Fallback LiDAR modality index if camera modality names are unavailable.",
    )
    parser.add_argument("--pelvis-index", type=int, default=0, help="Pelvis joint index for centering.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Odd temporal smoothing window per sequence. 1 disables smoothing.",
    )
    parser.add_argument(
        "--smooth-on",
        type=str,
        default="pred",
        choices=["pred", "gt", "both"],
        help="Which keypoints to smooth before metric computation.",
    )
    args = parser.parse_args()
    show_progress = not args.no_progress

    with open(args.pred_file, "rb") as f:
        data = pickle.load(f)

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

    pred_seq_cam_enc, pred_seq_meta = _build_sequence_reference_cameras(
        data=data,
        camera_key=args.pred_cameras_key,
        sample_ids=sample_ids,
        num_samples=num_samples,
        fallback_lidar_idx=args.lidar_modality_index,
        show_progress=show_progress,
    )
    gt_seq_cam_enc, gt_seq_meta = _build_sequence_reference_cameras(
        data=data,
        camera_key=args.gt_cameras_key,
        sample_ids=sample_ids,
        num_samples=num_samples,
        fallback_lidar_idx=args.lidar_modality_index,
        show_progress=show_progress,
    )

    pred_seq_extr = {
        seq: _pose_encoding_to_extrinsic(cam_enc, args.pose_encoding_type)
        for seq, cam_enc in pred_seq_cam_enc.items()
    }
    gt_seq_extr = {
        seq: _pose_encoding_to_extrinsic(cam_enc, args.pose_encoding_type)
        for seq, cam_enc in gt_seq_cam_enc.items()
    }

    pred_new = []
    gt_new = []
    kept_seq_names = []
    kept_frame_indices = []
    kept = 0
    dropped_no_seq_ref = 0
    for i in _iter_with_progress(
        range(num_samples),
        desc="transform keypoints",
        total=num_samples,
        enabled=show_progress,
    ):
        seq_name = _seq_name_from_sample_id(sample_ids[i] if i < len(sample_ids) else None)
        pred_extr = pred_seq_extr.get(seq_name, None)
        gt_extr = gt_seq_extr.get(seq_name, None)
        if pred_extr is None or gt_extr is None:
            dropped_no_seq_ref += 1
            continue
        pred_kps = _get_sample_keypoints(pred_all, i)
        gt_kps = _get_sample_keypoints(gt_all, i)
        if pred_kps is None or gt_kps is None:
            continue
        pred_kps = _to_numpy_maybe(pred_kps)
        gt_kps = _to_numpy_maybe(gt_kps)
        if pred_kps is None or gt_kps is None:
            continue
        if pred_kps.shape != gt_kps.shape:
            continue
        if pred_kps.ndim != 2 or pred_kps.shape[-1] != 3:
            continue
        if not np.isfinite(pred_kps).all() or not np.isfinite(gt_kps).all():
            continue
        pred_new.append(_transform_points(pred_kps.astype(np.float32), pred_extr))
        gt_new.append(_transform_points(gt_kps.astype(np.float32), gt_extr))
        kept_seq_names.append(seq_name)
        kept_frame_indices.append(_frame_index_from_sample_id(sample_ids[i] if i < len(sample_ids) else None))
        kept += 1

    if kept == 0:
        raise RuntimeError("No valid samples found for evaluation.")

    pred_new = np.stack(pred_new, axis=0)
    gt_new = np.stack(gt_new, axis=0)

    if args.smooth_window > 1:
        if args.smooth_on in {"pred", "both"}:
            pred_new = _smooth_temporal_by_sequence(
                pred_new,
                seq_names=kept_seq_names,
                frame_indices=kept_frame_indices,
                window=args.smooth_window,
            )
        if args.smooth_on in {"gt", "both"}:
            gt_new = _smooth_temporal_by_sequence(
                gt_new,
                seq_names=kept_seq_names,
                frame_indices=kept_frame_indices,
                window=args.smooth_window,
            )

    mpjpe_new = float(mpjpe_func(pred_new, gt_new, reduce=True))
    pampjpe_new = float(pampjpe_func(pred_new, gt_new, reduce=True))

    pelvis = int(args.pelvis_index)
    pred_centered = pred_new - pred_new[:, pelvis:pelvis + 1, :]
    gt_centered = gt_new - gt_new[:, pelvis:pelvis + 1, :]
    mpjpe_centered = float(mpjpe_func(pred_centered, gt_centered, reduce=True))

    print(f"[INFO] Loaded: {args.pred_file}")
    print(f"[INFO] Sequence reference cameras (pred): {len(pred_seq_extr)}")
    print(f"[INFO] Sequence reference cameras (gt):   {len(gt_seq_extr)}")
    example_seq = sorted(set(pred_seq_meta.keys()) & set(gt_seq_meta.keys()))
    if len(example_seq) > 0:
        seq0 = example_seq[0]
        p_idx, p_m = pred_seq_meta[seq0]
        g_idx, g_m = gt_seq_meta[seq0]
        print(
            f"[INFO] Example seq `{seq0}` refs: "
            f"pred(sample_idx={p_idx}, modality_idx={p_m}), "
            f"gt(sample_idx={g_idx}, modality_idx={g_m})"
        )
    if dropped_no_seq_ref > 0:
        print(f"[INFO] Dropped samples without per-sequence refs: {dropped_no_seq_ref}")
    print(f"[INFO] Smoothing: window={args.smooth_window}, target={args.smooth_on}")
    print(f"[INFO] Valid samples: {kept}/{num_samples}")
    print(f"[RESULT] MPJPE (fixed-sensor frame): {_format_mm(mpjpe_new)}")
    print(f"[RESULT] PA-MPJPE (fixed-sensor frame): {_format_mm(pampjpe_new)}")
    print(
        f"[RESULT] MPJPE (fixed-sensor frame, pelvis-centered @ joint {pelvis}): "
        f"{_format_mm(mpjpe_centered)}"
    )


if __name__ == "__main__":
    main()
