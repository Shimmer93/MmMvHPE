#!/usr/bin/env python3
import argparse
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
_PRECOMP_WORKER_CONTEXT: Optional[Dict[str, Any]] = None


def _set_eval_worker_context(context: Dict[str, Any]) -> None:
    global _EVAL_WORKER_CONTEXT
    _EVAL_WORKER_CONTEXT = context


def _init_eval_worker(context: Dict[str, Any]) -> None:
    _set_eval_worker_context(context)


def _set_precomp_worker_context(context: Dict[str, Any]) -> None:
    global _PRECOMP_WORKER_CONTEXT
    _PRECOMP_WORKER_CONTEXT = context


def _init_precomp_worker(context: Dict[str, Any]) -> None:
    _set_precomp_worker_context(context)


def _run_precomp_worker_pool(
    worker_fn,
    context: Dict[str, Any],
    num_samples: int,
    workers: int,
    show_progress: bool,
    desc: str,
):
    if workers <= 1:
        _set_precomp_worker_context(context)
        iterator = (worker_fn(i) for i in range(num_samples))
        for item in _iter_with_progress(iterator, desc=desc, total=num_samples, enabled=show_progress):
            yield item
        return

    chunksize = max(1, min(256, num_samples // max(workers * 4, 1)))
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_precomp_worker,
        initargs=(context,),
    ) as executor:
        iterator = executor.map(worker_fn, range(num_samples), chunksize=chunksize)
        for item in _iter_with_progress(iterator, desc=desc, total=num_samples, enabled=show_progress):
            yield item


def _seq_name_from_sample_id(sample_id: Optional[str]) -> str:
    return _seq_base_name_from_sample_id(sample_id)


def _sensor_combo_from_sample_id(sample_id: Optional[str]) -> Optional[str]:
    if sample_id is None:
        return None
    s = str(sample_id)
    parts = s.split("_")
    if len(parts) < 3:
        return None
    tokens: List[str] = []
    seen = set()
    for i in range(len(parts) - 2):
        mod = parts[i].strip().lower()
        sensor_family = parts[i + 1].strip().lower()
        sensor_idx = parts[i + 2].strip()
        if mod not in _KNOWN_SENSOR_MODALITY_PREFIXES:
            continue
        if (not sensor_family) or (not sensor_idx.isdigit()):
            continue
        token = f"{mod}_{sensor_family}_{sensor_idx}"
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    if len(tokens) == 0:
        return None
    return "+".join(tokens)


def _sensor_combo_from_stream_metadata_for_sample(data, sample_idx: int) -> Optional[str]:
    stream_modalities, stream_sensor_indices = _get_stream_metadata_for_sample(data, sample_idx)
    if stream_modalities is None or stream_sensor_indices is None:
        return None
    if len(stream_modalities) != len(stream_sensor_indices):
        return None
    if len(stream_modalities) == 0:
        return None
    tokens: List[str] = []
    seen = set()
    for m, idx in zip(stream_modalities, stream_sensor_indices):
        mod = str(m).strip().lower()
        if not mod:
            continue
        tok = f"{mod}_{int(idx)}"
        if tok in seen:
            continue
        seen.add(tok)
        tokens.append(tok)
    if len(tokens) == 0:
        return None
    return "+".join(tokens)


def _seq_group_name_for_sample(
    data,
    sample_ids: Sequence[str],
    sample_idx: int,
) -> str:
    sid = sample_ids[sample_idx] if sample_idx < len(sample_ids) else None
    base = _seq_base_name_from_sample_id(sid)
    combo_stream = _sensor_combo_from_stream_metadata_for_sample(data, sample_idx)
    combo_sample_id = _sensor_combo_from_sample_id(sid)
    if combo_stream is None and combo_sample_id is None:
        return base
    # Use both identifiers when available:
    # - stream metadata captures modality/sensor-index mapping
    # - sample_id sensor tags capture physical camera IDs
    # This avoids mixing different physical cameras that share the same
    # modality/sensor-index tuple.
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


def _build_seq_group_names(
    data,
    sample_ids: Sequence[str],
) -> List[str]:
    out: List[str] = []
    for i in range(len(sample_ids)):
        out.append(_seq_group_name_for_sample(data, sample_ids, i))
    return out


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
    if center.shape[0] != 3:
        return None
    if not np.isfinite(center).all():
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


def _is_stream_camera_key(camera_key: str) -> bool:
    return str(camera_key).endswith("_stream")


def _find_first_valid_lidar_camera(
    data,
    camera_key: str,
    num_samples: int,
    fallback_lidar_idx: Optional[int],
    lidar_sensor_idx: Optional[int],
    show_progress: bool = True,
) -> Tuple[np.ndarray, int, int]:
    cameras = data.get(camera_key, None)
    if cameras is None:
        raise ValueError(f"Missing `{camera_key}` in prediction file.")

    use_stream_index = _is_stream_camera_key(camera_key)
    for i in _iter_with_progress(
        range(num_samples),
        desc=f"search {camera_key}",
        total=num_samples,
        enabled=show_progress,
    ):
        lidar_idx = _get_camera_index(
            data=data,
            sample_idx=i,
            modality="lidar",
            fallback_idx=fallback_lidar_idx,
            use_stream_index=use_stream_index,
            sensor_idx=lidar_sensor_idx,
        )
        cam = _extract_camera_encoding(cameras, i, lidar_idx)
        center = _get_sample_lidar_center(data, i)
        cam = _inverse_lidar_camera_center(cam, center)
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
    lidar_sensor_idx: Optional[int],
    show_progress: bool = True,
):
    seq_to_camera, seq_to_meta, _ = _build_sequence_reference_cameras_for_modality_with_policy(
        data=data,
        camera_key=camera_key,
        sample_ids=sample_ids,
        num_samples=num_samples,
        modality="lidar",
        fallback_modality_idx=fallback_lidar_idx,
        sensor_idx=lidar_sensor_idx,
        ref_policy="first",
        pose_encoding_type="absT_quaR_FoV",
        ref_score_trans_scale=0.10,
        ref_score_rot_scale_deg=10.0,
        preferred_sensor_by_seq=None,
        show_progress=show_progress,
    )
    return seq_to_camera, seq_to_meta


_RGB_SENSOR_RE = re.compile(r"rgb_[^_]+_(\d+)")
_DEPTH_SENSOR_RE = re.compile(r"depth_[^_]+_(\d+)")
_LIDAR_SENSOR_RE = re.compile(r"lidar_[^_]+_(\d+)")


def _sensor_tag_from_sample_id(sample_id: Optional[str], modality: str) -> str:
    if sample_id is None:
        return "__default__"
    s = str(sample_id).lower()
    mod = str(modality).lower()
    patterns = []
    if mod == "rgb":
        patterns = [_RGB_SENSOR_RE]
    elif mod == "lidar":
        patterns = [_LIDAR_SENSOR_RE, _DEPTH_SENSOR_RE]
    else:
        patterns = [re.compile(rf"{re.escape(mod)}_[^_]+_(\d+)")]
    for pat in patterns:
        m = pat.search(s)
        if m is not None:
            return str(m.group(1))
    return "__default__"


def _build_sequence_reference_camera_candidates_for_modality(
    data,
    camera_key: str,
    sample_ids: Sequence[str],
    num_samples: int,
    modality: str,
    fallback_modality_idx: Optional[int],
    sensor_idx: Optional[int],
    show_progress: bool = True,
):
    cameras = data.get(camera_key, None)
    if cameras is None:
        raise ValueError(f"Missing `{camera_key}` in prediction file.")

    mod = str(modality).lower()
    use_stream_index = _is_stream_camera_key(camera_key)
    seq_to_sensor_camera: Dict[str, Dict[str, np.ndarray]] = {}
    seq_to_sensor_meta: Dict[str, Dict[str, Tuple[int, int]]] = {}
    iterator = _iter_with_progress(
        range(num_samples),
        desc=f"build seq refs {camera_key}:{mod}",
        total=num_samples,
        enabled=show_progress,
    )
    for i in iterator:
        sid = sample_ids[i] if i < len(sample_ids) else None
        seq_name = _seq_group_name_for_sample(data, sample_ids, i)
        sensor_tag = _sensor_tag_from_sample_id(sid, mod)

        camera_idx = _get_camera_index(
            data=data,
            sample_idx=i,
            modality=mod,
            fallback_idx=fallback_modality_idx,
            use_stream_index=use_stream_index,
            sensor_idx=sensor_idx,
        )
        cam = _extract_camera_encoding(cameras, i, camera_idx)
        if mod == "lidar":
            center = _get_sample_lidar_center(data, i)
            cam = _inverse_lidar_camera_center(cam, center)
        if cam is None or not np.isfinite(cam).all():
            continue

        per_seq = seq_to_sensor_camera.setdefault(seq_name, {})
        per_seq_meta = seq_to_sensor_meta.setdefault(seq_name, {})
        if sensor_tag in per_seq:
            continue
        per_seq[sensor_tag] = cam.astype(np.float32)
        per_seq_meta[sensor_tag] = (i, int(0 if camera_idx is None else camera_idx))

    if len(seq_to_sensor_camera) == 0:
        raise ValueError(f"Could not find any finite `{mod}` camera in `{camera_key}`.")
    return seq_to_sensor_camera, seq_to_sensor_meta


def _build_sequence_extrinsics_for_modality(
    data,
    camera_key: str,
    sample_ids: Sequence[str],
    num_samples: int,
    modality: str,
    fallback_modality_idx: Optional[int],
    sensor_idx: Optional[int],
    pose_encoding_type: str,
    show_progress: bool = True,
) -> Dict[str, List[np.ndarray]]:
    cameras = data.get(camera_key, None)
    if cameras is None:
        raise ValueError(f"Missing `{camera_key}` in prediction file.")

    mod = str(modality).lower()
    use_stream_index = _is_stream_camera_key(camera_key)
    seq_to_extrs: Dict[str, List[np.ndarray]] = {}
    iterator = _iter_with_progress(
        range(num_samples),
        desc=f"decode seq refs {camera_key}:{mod}",
        total=num_samples,
        enabled=show_progress,
    )
    for i in iterator:
        seq_name = _seq_group_name_for_sample(data, sample_ids, i)
        camera_idx = _get_camera_index(
            data=data,
            sample_idx=i,
            modality=mod,
            fallback_idx=fallback_modality_idx,
            use_stream_index=use_stream_index,
            sensor_idx=sensor_idx,
        )
        cam = _extract_camera_encoding(cameras, i, camera_idx)
        if mod == "lidar":
            center = _get_sample_lidar_center(data, i)
            cam = _inverse_lidar_camera_center(cam, center)
        if cam is None or not np.isfinite(cam).all():
            continue
        try:
            extr = _pose_encoding_to_extrinsic(cam, pose_encoding_type)
        except Exception:
            continue
        if not np.isfinite(extr).all():
            continue
        seq_to_extrs.setdefault(seq_name, []).append(extr.astype(np.float32))
    return seq_to_extrs


def _reference_camera_stability_score(
    ref_extr: np.ndarray,
    seq_extrinsics: Sequence[np.ndarray],
    trans_scale: float,
    rot_scale_deg: float,
) -> float:
    if len(seq_extrinsics) == 0:
        return float("inf")
    ref_r = np.asarray(ref_extr, dtype=np.float64)[:, :3]
    ref_t = np.asarray(ref_extr, dtype=np.float64)[:, 3]
    rot_errs = []
    trans_errs = []
    for extr in seq_extrinsics:
        e = np.asarray(extr, dtype=np.float64).reshape(3, 4)
        rot_errs.append(_rotation_angle_deg(ref_r, e[:, :3]))
        trans_errs.append(float(np.linalg.norm(ref_t - e[:, 3])))
    med_rot = float(np.median(np.asarray(rot_errs, dtype=np.float64)))
    med_trans = float(np.median(np.asarray(trans_errs, dtype=np.float64)))
    return med_rot / max(float(rot_scale_deg), 1e-8) + med_trans / max(float(trans_scale), 1e-8)


def _medoid_camera_index_from_extrinsics(
    extrinsics: Sequence[np.ndarray],
    trans_scale: float,
    rot_scale_deg: float,
) -> int:
    if len(extrinsics) == 0:
        raise ValueError("Empty extrinsics list for medoid selection.")
    if len(extrinsics) == 1:
        return 0

    e = np.asarray(extrinsics, dtype=np.float64).reshape(len(extrinsics), 3, 4)
    r_all = e[:, :, :3]
    t_all = e[:, :, 3]

    t_scale = max(float(trans_scale), 1e-8)
    r_scale = max(float(rot_scale_deg), 1e-8)

    best_idx = 0
    best_score = float("inf")
    r_all_t = np.transpose(r_all, (0, 2, 1))
    for i in range(e.shape[0]):
        rel = np.einsum("ij,njk->nik", r_all[i], r_all_t)
        trace = np.einsum("nii->n", rel)
        cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        rot_deg = np.degrees(np.arccos(cos_theta))
        trans = np.linalg.norm(t_all - t_all[i], axis=1)
        d = rot_deg / r_scale + trans / t_scale
        finite = np.isfinite(d)
        if not np.any(finite):
            continue
        score = float(np.median(d[finite]))
        if score < best_score:
            best_score = score
            best_idx = i
    return int(best_idx)


def _build_sequence_reference_cameras_for_modality_with_policy(
    data,
    camera_key: str,
    sample_ids: Sequence[str],
    num_samples: int,
    modality: str,
    fallback_modality_idx: Optional[int],
    sensor_idx: Optional[int],
    ref_policy: str,
    pose_encoding_type: str,
    ref_score_trans_scale: float,
    ref_score_rot_scale_deg: float,
    preferred_sensor_by_seq: Optional[Dict[str, str]],
    show_progress: bool = True,
):
    mod = str(modality).lower()
    policy = str(ref_policy).lower()
    if policy not in {"first", "best_sensor"}:
        raise ValueError(f"Unknown seq ref policy `{ref_policy}`.")

    seq_to_sensor_camera, seq_to_sensor_meta = _build_sequence_reference_camera_candidates_for_modality(
        data=data,
        camera_key=camera_key,
        sample_ids=sample_ids,
        num_samples=num_samples,
        modality=mod,
        fallback_modality_idx=fallback_modality_idx,
        sensor_idx=sensor_idx,
        show_progress=show_progress,
    )

    chosen_sensor_by_seq: Dict[str, str] = {}
    seq_to_camera: Dict[str, np.ndarray] = {}
    seq_to_meta: Dict[str, Tuple[int, int]] = {}

    if preferred_sensor_by_seq is not None:
        for seq_name, sensor_to_cam in seq_to_sensor_camera.items():
            chosen = preferred_sensor_by_seq.get(seq_name, None)
            if chosen not in sensor_to_cam:
                chosen = next(iter(sensor_to_cam.keys()))
            chosen_sensor_by_seq[seq_name] = str(chosen)
            seq_to_camera[seq_name] = sensor_to_cam[str(chosen)]
            seq_to_meta[seq_name] = seq_to_sensor_meta[seq_name][str(chosen)]
        return seq_to_camera, seq_to_meta, chosen_sensor_by_seq

    if policy == "first":
        for seq_name, sensor_to_cam in seq_to_sensor_camera.items():
            chosen = next(iter(sensor_to_cam.keys()))
            chosen_sensor_by_seq[seq_name] = str(chosen)
            seq_to_camera[seq_name] = sensor_to_cam[str(chosen)]
            seq_to_meta[seq_name] = seq_to_sensor_meta[seq_name][str(chosen)]
        return seq_to_camera, seq_to_meta, chosen_sensor_by_seq

    seq_to_extrs = _build_sequence_extrinsics_for_modality(
        data=data,
        camera_key=camera_key,
        sample_ids=sample_ids,
        num_samples=num_samples,
        modality=mod,
        fallback_modality_idx=fallback_modality_idx,
        sensor_idx=sensor_idx,
        pose_encoding_type=pose_encoding_type,
        show_progress=show_progress,
    )
    for seq_name, sensor_to_cam in seq_to_sensor_camera.items():
        best_sensor = None
        best_score = float("inf")
        seq_extrs = seq_to_extrs.get(seq_name, [])
        for sensor_tag, cam_enc in sensor_to_cam.items():
            try:
                ref_extr = _pose_encoding_to_extrinsic(cam_enc, pose_encoding_type)
            except Exception:
                continue
            score = _reference_camera_stability_score(
                ref_extr=ref_extr,
                seq_extrinsics=seq_extrs,
                trans_scale=float(ref_score_trans_scale),
                rot_scale_deg=float(ref_score_rot_scale_deg),
            )
            if score < best_score:
                best_score = score
                best_sensor = sensor_tag
        if best_sensor is None:
            best_sensor = next(iter(sensor_to_cam.keys()))
        chosen_sensor_by_seq[seq_name] = str(best_sensor)
        seq_to_camera[seq_name] = sensor_to_cam[str(best_sensor)]
        seq_to_meta[seq_name] = seq_to_sensor_meta[seq_name][str(best_sensor)]
    return seq_to_camera, seq_to_meta, chosen_sensor_by_seq


def _build_paired_sequence_reference_cameras_for_modality(
    data,
    pred_camera_key: str,
    gt_camera_key: str,
    sample_ids: Sequence[str],
    num_samples: int,
    modality: str,
    fallback_modality_idx: Optional[int],
    sensor_idx: Optional[int],
    ref_policy: str,
    pose_encoding_type: str,
    ref_score_trans_scale: float,
    ref_score_rot_scale_deg: float,
    show_progress: bool = True,
):
    pred_seq_cam_enc, pred_seq_meta, chosen_sensor_by_seq = _build_sequence_reference_cameras_for_modality_with_policy(
        data=data,
        camera_key=pred_camera_key,
        sample_ids=sample_ids,
        num_samples=num_samples,
        modality=modality,
        fallback_modality_idx=fallback_modality_idx,
        sensor_idx=sensor_idx,
        ref_policy=ref_policy,
        pose_encoding_type=pose_encoding_type,
        ref_score_trans_scale=ref_score_trans_scale,
        ref_score_rot_scale_deg=ref_score_rot_scale_deg,
        preferred_sensor_by_seq=None,
        show_progress=show_progress,
    )
    gt_seq_cam_enc, gt_seq_meta, _ = _build_sequence_reference_cameras_for_modality_with_policy(
        data=data,
        camera_key=gt_camera_key,
        sample_ids=sample_ids,
        num_samples=num_samples,
        modality=modality,
        fallback_modality_idx=fallback_modality_idx,
        sensor_idx=sensor_idx,
        ref_policy="first",
        pose_encoding_type=pose_encoding_type,
        ref_score_trans_scale=ref_score_trans_scale,
        ref_score_rot_scale_deg=ref_score_rot_scale_deg,
        preferred_sensor_by_seq=chosen_sensor_by_seq,
        show_progress=show_progress,
    )
    return pred_seq_cam_enc, pred_seq_meta, gt_seq_cam_enc, gt_seq_meta, chosen_sensor_by_seq


def _build_sequence_reference_cameras_for_modality(
    data,
    camera_key: str,
    sample_ids: Sequence[str],
    num_samples: int,
    modality: str,
    fallback_modality_idx: Optional[int],
    sensor_idx: Optional[int],
    show_progress: bool = True,
):
    seq_to_camera, seq_to_meta, _ = _build_sequence_reference_cameras_for_modality_with_policy(
        data=data,
        camera_key=camera_key,
        sample_ids=sample_ids,
        num_samples=num_samples,
        modality=modality,
        fallback_modality_idx=fallback_modality_idx,
        sensor_idx=sensor_idx,
        ref_policy="first",
        pose_encoding_type="absT_quaR_FoV",
        ref_score_trans_scale=0.10,
        ref_score_rot_scale_deg=10.0,
        preferred_sensor_by_seq=None,
        show_progress=show_progress,
    )
    return seq_to_camera, seq_to_meta


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
        x, y, z, w = (q / qn).tolist()  # XYZW, scalar-last
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


def _to_homogeneous(extrinsic: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float32)
    out[:3, :4] = extrinsic.astype(np.float32)
    return out


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3).tolist()
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def _so3_exp(rotvec: np.ndarray) -> np.ndarray:
    w = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    wx = _skew(w)
    i3 = np.eye(3, dtype=np.float64)
    if theta < 1e-12:
        return i3 + wx
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return i3 + a * wx + b * (wx @ wx)


def _so3_log(rot: np.ndarray) -> np.ndarray:
    r = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(r))
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-12:
        return np.array(
            [
                0.5 * (r[2, 1] - r[1, 2]),
                0.5 * (r[0, 2] - r[2, 0]),
                0.5 * (r[1, 0] - r[0, 1]),
            ],
            dtype=np.float64,
        )
    s = np.sin(theta)
    if abs(s) < 1e-12:
        vals, vecs = np.linalg.eig(r)
        idx = int(np.argmin(np.abs(vals - 1.0)))
        axis = np.real(vecs[:, idx]).astype(np.float64)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        return axis * theta
    scale = theta / (2.0 * s)
    return np.array(
        [
            scale * (r[2, 1] - r[1, 2]),
            scale * (r[0, 2] - r[2, 0]),
            scale * (r[1, 0] - r[0, 1]),
        ],
        dtype=np.float64,
    )


def _huber_weights(norms: np.ndarray, delta: float) -> np.ndarray:
    d = max(float(delta), 1e-8)
    n = np.asarray(norms, dtype=np.float64)
    w = np.ones_like(n, dtype=np.float64)
    mask = n > d
    if np.any(mask):
        w[mask] = d / np.maximum(n[mask], 1e-12)
    return w


def _robust_mean_vectors_huber(
    vectors: np.ndarray,
    huber_delta: float,
    max_iters: int,
) -> np.ndarray:
    vecs = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    if vecs.shape[0] == 1:
        return vecs[0]
    mu = np.median(vecs, axis=0)
    for _ in range(max(int(max_iters), 1)):
        res = vecs - mu[None, :]
        norms = np.linalg.norm(res, axis=1)
        w = _huber_weights(norms, huber_delta)
        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 1e-12:
            break
        mu_new = np.sum(vecs * w[:, None], axis=0) / w_sum
        if float(np.linalg.norm(mu_new - mu)) < 1e-9:
            mu = mu_new
            break
        mu = mu_new
    return mu


def _robust_mean_rotations_huber(
    rotations: Sequence[np.ndarray],
    huber_delta_deg: float,
    max_iters: int,
) -> np.ndarray:
    rs = [np.asarray(r, dtype=np.float64).reshape(3, 3) for r in rotations]
    if len(rs) == 1:
        return rs[0]
    r_mean = rs[0]
    delta_rad = np.deg2rad(max(float(huber_delta_deg), 1e-6))
    for _ in range(max(int(max_iters), 1)):
        errs = []
        for r in rs:
            errs.append(_so3_log(r_mean.T @ r))
        err_arr = np.stack(errs, axis=0)
        norms = np.linalg.norm(err_arr, axis=1)
        w = _huber_weights(norms, delta_rad)
        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 1e-12:
            break
        step = np.sum(err_arr * w[:, None], axis=0) / w_sum
        step_norm = float(np.linalg.norm(step))
        if step_norm < 1e-9:
            break
        r_mean = r_mean @ _so3_exp(step)
    return r_mean


def _robust_mean_extrinsics_huber(
    extrinsics: Sequence[np.ndarray],
    huber_trans_delta: float,
    huber_rot_delta_deg: float,
    max_iters: int,
) -> np.ndarray:
    exts = [np.asarray(e, dtype=np.float64).reshape(3, 4) for e in extrinsics]
    if len(exts) == 0:
        raise ValueError("Cannot compute robust mean for empty extrinsic list.")
    if len(exts) == 1:
        return exts[0].astype(np.float32)

    rotations = [e[:, :3] for e in exts]
    translations = np.stack([e[:, 3] for e in exts], axis=0)
    r_mean = _robust_mean_rotations_huber(
        rotations=rotations,
        huber_delta_deg=huber_rot_delta_deg,
        max_iters=max_iters,
    )
    t_mean = _robust_mean_vectors_huber(
        vectors=translations,
        huber_delta=huber_trans_delta,
        max_iters=max_iters,
    )
    out = np.concatenate([r_mean, t_mean.reshape(3, 1)], axis=1)
    return out.astype(np.float32)


def _parse_modalities_list(text: str) -> List[str]:
    parts = [p.strip().lower() for p in str(text).split(",")]
    return [p for p in parts if p]


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
                f"Invalid --sensor-index-by-modality entry `{part}`. Expected format `modality:index`."
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
        raise ValueError(
            f"Requested camera key `{requested_key}` is not in prediction file keys."
        )
    for key in fallback_keys:
        if key in data:
            return key
    raise ValueError(
        f"None of the camera keys exist in prediction file: requested={requested_key}, "
        f"fallbacks={list(fallback_keys)}."
    )


def _rotation_angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    r = np.asarray(r_a, dtype=np.float64) @ np.asarray(r_b, dtype=np.float64).T
    trace = float(np.trace(r))
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _stability_to_score(
    trans_std: float,
    rot_std_deg: float,
    trans_tau: float,
    rot_tau_deg: float,
) -> float:
    t_tau = max(float(trans_tau), 1e-8)
    r_tau = max(float(rot_tau_deg), 1e-8)
    score = np.exp(-(float(trans_std) / t_tau + float(rot_std_deg) / r_tau))
    return float(np.clip(score, 1e-8, 1.0))


def _temporal_reliability_seqfixed_sample_worker(sample_idx: int):
    if _PRECOMP_WORKER_CONTEXT is None:
        raise RuntimeError("Precompute worker context is not initialized.")
    ctx = _PRECOMP_WORKER_CONTEXT
    data = ctx["data"]
    cameras = ctx["cameras"]
    use_stream_index = bool(ctx["use_stream_index"])
    target_mod = str(ctx["target_mod"])
    mods = list(ctx["mods"])
    modality_fallback_indices = ctx["modality_fallback_indices"]
    modality_sensor_indices = ctx["modality_sensor_indices"]
    pose_encoding_type = str(ctx["pose_encoding_type"])
    sample_ids = ctx["sample_ids"]

    seq_name = _seq_group_name_for_sample(data, sample_ids, sample_idx)
    target_idx = _get_camera_index(
        data=data,
        sample_idx=sample_idx,
        modality=target_mod,
        fallback_idx=modality_fallback_indices.get(target_mod, None),
        use_stream_index=use_stream_index,
        sensor_idx=modality_sensor_indices.get(target_mod, 0),
    )
    target_cam = _extract_camera_encoding(cameras, sample_idx, target_idx)
    if target_mod == "lidar":
        center = _get_sample_lidar_center(data, sample_idx)
        target_cam = _inverse_lidar_camera_center(target_cam, center)
    if target_cam is None:
        return seq_name, []
    target_cam = np.asarray(target_cam, dtype=np.float32).reshape(-1)
    if target_cam.shape[0] < 9 or (not np.isfinite(target_cam).all()):
        return seq_name, []
    try:
        target_extr = _pose_encoding_to_extrinsic(target_cam, pose_encoding_type)
    except Exception:
        return seq_name, []
    if not np.isfinite(target_extr).all():
        return seq_name, []
    h_target = _to_homogeneous(target_extr)

    entries = []
    for mod in mods:
        m_idx = _get_camera_index(
            data=data,
            sample_idx=sample_idx,
            modality=mod,
            fallback_idx=modality_fallback_indices.get(mod, None),
            use_stream_index=use_stream_index,
            sensor_idx=modality_sensor_indices.get(mod, 0),
        )
        cam = _extract_camera_encoding(cameras, sample_idx, m_idx)
        if mod == "lidar":
            center = _get_sample_lidar_center(data, sample_idx)
            cam = _inverse_lidar_camera_center(cam, center)
        if cam is None:
            continue
        cam = np.asarray(cam, dtype=np.float32).reshape(-1)
        if cam.shape[0] < 9 or (not np.isfinite(cam).all()):
            continue
        try:
            extr = _pose_encoding_to_extrinsic(cam, pose_encoding_type)
        except Exception:
            continue
        if not np.isfinite(extr).all():
            continue
        h_mod = _to_homogeneous(extr)
        try:
            h_rel = h_target @ np.linalg.inv(h_mod)
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(h_rel).all():
            continue

        entries.append(
            (
                mod,
                h_rel[:3, :3].astype(np.float64),
                h_rel[:3, 3].astype(np.float64),
            )
        )
    return seq_name, entries


def _decode_pred_extrinsics_sample_worker(sample_idx: int):
    if _PRECOMP_WORKER_CONTEXT is None:
        raise RuntimeError("Precompute worker context is not initialized.")
    ctx = _PRECOMP_WORKER_CONTEXT
    data = ctx["data"]
    cameras = ctx["cameras"]
    use_stream_index = bool(ctx["use_stream_index"])
    mods = list(ctx["mods"])
    modality_fallback_indices = ctx["modality_fallback_indices"]
    modality_sensor_indices = ctx["modality_sensor_indices"]
    pose_encoding_type = str(ctx["pose_encoding_type"])

    out: Dict[str, np.ndarray] = {}
    for mod in mods:
        camera_idx = _get_camera_index(
            data=data,
            sample_idx=sample_idx,
            modality=mod,
            fallback_idx=modality_fallback_indices.get(mod, None),
            use_stream_index=use_stream_index,
            sensor_idx=modality_sensor_indices.get(mod, 0),
        )
        cam = _extract_camera_encoding(cameras, sample_idx, camera_idx)
        if mod == "lidar":
            center = _get_sample_lidar_center(data, sample_idx)
            cam = _inverse_lidar_camera_center(cam, center)
        if cam is None:
            continue
        cam = np.asarray(cam, dtype=np.float32).reshape(-1)
        if cam.shape[0] < 9 or (not np.isfinite(cam).all()):
            continue
        try:
            extr = _pose_encoding_to_extrinsic(cam, pose_encoding_type)
        except Exception:
            continue
        if not np.isfinite(extr).all():
            continue
        out[mod] = extr.astype(np.float32)
    return out


def _estimate_seq_rig_rel_sample_worker(sample_idx: int):
    if _PRECOMP_WORKER_CONTEXT is None:
        raise RuntimeError("Precompute worker context is not initialized.")
    ctx = _PRECOMP_WORKER_CONTEXT
    sample_ids = ctx["sample_ids"]
    seq_group_names = ctx.get("seq_group_names", None)
    pred_extr_by_modality = ctx["pred_extr_by_modality"]
    target_mod = str(ctx["target_mod"])
    mods = list(ctx["mods"])
    num_samples = int(ctx["num_samples"])

    if seq_group_names is not None and sample_idx < len(seq_group_names):
        seq_name = str(seq_group_names[sample_idx])
    else:
        seq_name = _seq_name_from_sample_id(sample_ids[sample_idx] if sample_idx < len(sample_ids) else None)
    target_extr = pred_extr_by_modality.get(target_mod, [None] * num_samples)[sample_idx]
    if target_extr is None:
        return []
    h_target = _to_homogeneous(target_extr)

    out = []
    for mod in mods:
        mod_extr = pred_extr_by_modality.get(mod, [None] * num_samples)[sample_idx]
        if mod_extr is None:
            continue
        h_mod = _to_homogeneous(mod_extr)
        try:
            h_rel = h_target @ np.linalg.inv(h_mod)
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(h_rel).all():
            continue
        out.append((mod, seq_name, h_rel[:3, :4].astype(np.float32)))
    return out


def _temporal_reliability_projected_sample_worker(sample_idx: int):
    if _PRECOMP_WORKER_CONTEXT is None:
        raise RuntimeError("Precompute worker context is not initialized.")
    ctx = _PRECOMP_WORKER_CONTEXT
    sample_ids = ctx["sample_ids"]
    seq_group_names = ctx.get("seq_group_names", None)
    pred_extr_by_modality = ctx["pred_extr_by_modality"]
    seq_rig_extr_by_modality = ctx["seq_rig_extr_by_modality"]
    target_mod = str(ctx["target_mod"])
    mods = list(ctx["mods"])
    num_samples = int(ctx["num_samples"])

    if seq_group_names is not None and sample_idx < len(seq_group_names):
        seq_name = str(seq_group_names[sample_idx])
    else:
        seq_name = _seq_name_from_sample_id(sample_ids[sample_idx] if sample_idx < len(sample_ids) else None)
    target_extr = pred_extr_by_modality.get(target_mod, [None] * num_samples)[sample_idx]
    if target_extr is None:
        return seq_name, []
    h_target = _to_homogeneous(target_extr)

    out = []
    for mod in mods:
        mod_extr = pred_extr_by_modality.get(mod, [None] * num_samples)[sample_idx]
        rig_extr = seq_rig_extr_by_modality.get(mod, {}).get(seq_name, None)
        if mod_extr is None or rig_extr is None:
            continue
        h_proj = _to_homogeneous(rig_extr) @ _to_homogeneous(mod_extr)
        if not np.isfinite(h_proj).all():
            continue
        try:
            h_res = h_proj @ np.linalg.inv(h_target)
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(h_res).all():
            continue
        out.append((mod, h_res[:3, :3].astype(np.float64), h_res[:3, 3].astype(np.float64)))
    return seq_name, out


def _build_temporal_reliability_by_sequence(
    data,
    camera_key: str,
    sample_ids: Sequence[str],
    num_samples: int,
    modalities: Sequence[str],
    target_modality: str,
    modality_fallback_indices: Dict[str, Optional[int]],
    modality_sensor_indices: Dict[str, int],
    pose_encoding_type: str,
    trans_tau: float,
    rot_tau_deg: float,
    show_progress: bool = True,
    workers: int = 1,
) -> Dict[str, Dict[str, float]]:
    cameras = data.get(camera_key, None)
    if cameras is None:
        return {}
    use_stream_index = _is_stream_camera_key(camera_key)
    target_mod = str(target_modality).lower()
    mods = [str(m).lower() for m in modalities]

    stats_by_mod: Dict[str, Dict[str, dict]] = {m: {} for m in mods}
    worker_context = {
        "data": data,
        "cameras": cameras,
        "use_stream_index": use_stream_index,
        "target_mod": target_mod,
        "mods": mods,
        "modality_fallback_indices": modality_fallback_indices,
        "modality_sensor_indices": modality_sensor_indices,
        "pose_encoding_type": pose_encoding_type,
        "sample_ids": sample_ids,
    }
    for seq_name, entries in _run_precomp_worker_pool(
        worker_fn=_temporal_reliability_seqfixed_sample_worker,
        context=worker_context,
        num_samples=num_samples,
        workers=workers,
        show_progress=show_progress,
        desc=f"temporal reliability {camera_key}",
    ):
        for mod, r, t in entries:
            seq_stats = stats_by_mod[mod].setdefault(
                seq_name,
                {
                    "n": 0,
                    "t_mean": np.zeros(3, dtype=np.float64),
                    "t_m2": np.zeros(3, dtype=np.float64),
                    "r0": None,
                    "a_mean": 0.0,
                    "a_m2": 0.0,
                },
            )
            seq_stats["n"] += 1
            n = int(seq_stats["n"])

            delta_t = t - seq_stats["t_mean"]
            seq_stats["t_mean"] += delta_t / n
            delta_t2 = t - seq_stats["t_mean"]
            seq_stats["t_m2"] += delta_t * delta_t2

            if seq_stats["r0"] is None:
                seq_stats["r0"] = r.copy()
                angle = 0.0
            else:
                angle = _rotation_angle_deg(r, seq_stats["r0"])
            delta_a = angle - seq_stats["a_mean"]
            seq_stats["a_mean"] += delta_a / n
            delta_a2 = angle - seq_stats["a_mean"]
            seq_stats["a_m2"] += delta_a * delta_a2

    out: Dict[str, Dict[str, float]] = {m: {} for m in mods}
    for mod, per_seq in stats_by_mod.items():
        for seq_name, st in per_seq.items():
            n = int(st["n"])
            if n <= 1:
                out[mod][seq_name] = 1.0
                continue
            t_var = st["t_m2"] / max(n - 1, 1)
            t_std = float(np.mean(np.sqrt(np.maximum(t_var, 0.0))))
            a_var = float(st["a_m2"]) / max(n - 1, 1)
            a_std = float(np.sqrt(max(a_var, 0.0)))
            out[mod][seq_name] = _stability_to_score(
                trans_std=t_std,
                rot_std_deg=a_std,
                trans_tau=trans_tau,
                rot_tau_deg=rot_tau_deg,
            )
    return out


def _build_pred_extrinsics_by_sample_for_modalities(
    data,
    camera_key: str,
    sample_ids: Sequence[str],
    num_samples: int,
    modalities: Sequence[str],
    modality_fallback_indices: Dict[str, Optional[int]],
    modality_sensor_indices: Dict[str, int],
    pose_encoding_type: str,
    show_progress: bool = True,
    workers: int = 1,
) -> Dict[str, List[Optional[np.ndarray]]]:
    cameras = data.get(camera_key, None)
    if cameras is None:
        raise ValueError(f"Missing `{camera_key}` in prediction file.")
    use_stream_index = _is_stream_camera_key(camera_key)
    mods = [str(m).lower() for m in modalities]
    out: Dict[str, List[Optional[np.ndarray]]] = {m: [None] * num_samples for m in mods}
    worker_context = {
        "data": data,
        "cameras": cameras,
        "use_stream_index": use_stream_index,
        "mods": mods,
        "modality_fallback_indices": modality_fallback_indices,
        "modality_sensor_indices": modality_sensor_indices,
        "pose_encoding_type": pose_encoding_type,
    }
    for i, sample_extr_by_mod in enumerate(
        _run_precomp_worker_pool(
            worker_fn=_decode_pred_extrinsics_sample_worker,
            context=worker_context,
            num_samples=num_samples,
            workers=workers,
            show_progress=show_progress,
            desc=f"decode cameras {camera_key}",
        )
    ):
        for mod, extr in sample_extr_by_mod.items():
            if mod in out:
                out[mod][i] = extr
    return out


def _estimate_seq_rig_transforms_from_predicted_cameras(
    sample_ids: Sequence[str],
    seq_group_names: Optional[Sequence[str]],
    pred_extr_by_modality: Dict[str, List[Optional[np.ndarray]]],
    target_modality: str,
    modalities: Sequence[str],
    huber_trans_delta: float,
    huber_rot_delta_deg: float,
    max_iters: int,
    show_progress: bool = True,
    workers: int = 1,
) -> Dict[str, Dict[str, np.ndarray]]:
    target_mod = str(target_modality).lower()
    if target_mod not in pred_extr_by_modality:
        raise ValueError(f"Target modality `{target_mod}` missing from predicted extrinsics.")
    mods = [str(m).lower() for m in modalities]
    rel_by_mod_seq: Dict[str, Dict[str, List[np.ndarray]]] = {m: {} for m in mods}
    num_samples = len(sample_ids)

    worker_context = {
        "sample_ids": sample_ids,
        "seq_group_names": seq_group_names,
        "pred_extr_by_modality": pred_extr_by_modality,
        "target_mod": target_mod,
        "mods": mods,
        "num_samples": num_samples,
    }
    for entries in _run_precomp_worker_pool(
        worker_fn=_estimate_seq_rig_rel_sample_worker,
        context=worker_context,
        num_samples=num_samples,
        workers=workers,
        show_progress=show_progress,
        desc="estimate rig transforms",
    ):
        for mod, seq_name, rel_extr in entries:
            rel_by_mod_seq.setdefault(mod, {}).setdefault(seq_name, []).append(rel_extr)

    out: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in mods}
    for mod in mods:
        for seq_name, rel_list in rel_by_mod_seq.get(mod, {}).items():
            if len(rel_list) == 0:
                continue
            out[mod][seq_name] = _robust_mean_extrinsics_huber(
                extrinsics=rel_list,
                huber_trans_delta=huber_trans_delta,
                huber_rot_delta_deg=huber_rot_delta_deg,
                max_iters=max_iters,
            )
    return out


def _estimate_seq_reference_extrinsics_from_sample_extrinsics(
    sample_ids: Sequence[str],
    sample_extrinsics: Sequence[Optional[np.ndarray]],
    seq_group_names: Optional[Sequence[str]],
    huber_trans_delta: float,
    huber_rot_delta_deg: float,
    max_iters: int,
    show_progress: bool = True,
    desc: str = "estimate seq refs",
) -> Dict[str, np.ndarray]:
    seq_to_extrs: Dict[str, List[np.ndarray]] = {}
    iterator = _iter_with_progress(
        range(len(sample_extrinsics)),
        desc=desc,
        total=len(sample_extrinsics),
        enabled=show_progress,
    )
    for i in iterator:
        extr = sample_extrinsics[i]
        if extr is None:
            continue
        if seq_group_names is not None and i < len(seq_group_names):
            seq_name = str(seq_group_names[i])
        else:
            sid = sample_ids[i] if i < len(sample_ids) else None
            seq_name = _seq_name_from_sample_id(sid)
        seq_to_extrs.setdefault(seq_name, []).append(np.asarray(extr, dtype=np.float32))

    if len(seq_to_extrs) == 0:
        raise ValueError("No finite per-sample extrinsics available for sequence reference estimation.")

    out: Dict[str, np.ndarray] = {}
    for seq_name, exts in seq_to_extrs.items():
        out[seq_name] = _robust_mean_extrinsics_huber(
            extrinsics=exts,
            huber_trans_delta=huber_trans_delta,
            huber_rot_delta_deg=huber_rot_delta_deg,
            max_iters=max_iters,
        )
    return out


def _estimate_seq_reference_extrinsics_by_medoid(
    sample_ids: Sequence[str],
    sample_extrinsics: Sequence[Optional[np.ndarray]],
    seq_group_names: Optional[Sequence[str]],
    trans_scale: float,
    rot_scale_deg: float,
    show_progress: bool = True,
    desc: str = "estimate seq refs (medoid)",
) -> Dict[str, np.ndarray]:
    seq_to_extrs: Dict[str, List[np.ndarray]] = {}
    iterator = _iter_with_progress(
        range(len(sample_extrinsics)),
        desc=desc,
        total=len(sample_extrinsics),
        enabled=show_progress,
    )
    for i in iterator:
        extr = sample_extrinsics[i]
        if extr is None:
            continue
        if seq_group_names is not None and i < len(seq_group_names):
            seq_name = str(seq_group_names[i])
        else:
            sid = sample_ids[i] if i < len(sample_ids) else None
            seq_name = _seq_name_from_sample_id(sid)
        seq_to_extrs.setdefault(seq_name, []).append(np.asarray(extr, dtype=np.float32))

    if len(seq_to_extrs) == 0:
        raise ValueError("No finite per-sample extrinsics available for sequence reference estimation.")

    out: Dict[str, np.ndarray] = {}
    for seq_name, exts in seq_to_extrs.items():
        m_idx = _medoid_camera_index_from_extrinsics(
            extrinsics=exts,
            trans_scale=float(trans_scale),
            rot_scale_deg=float(rot_scale_deg),
        )
        out[seq_name] = np.asarray(exts[m_idx], dtype=np.float32)
    return out


def _estimate_seq_reference_extrinsics_with_policy(
    ref_policy: str,
    sample_ids: Sequence[str],
    sample_extrinsics: Sequence[Optional[np.ndarray]],
    seq_group_names: Optional[Sequence[str]],
    huber_trans_delta: float,
    huber_rot_delta_deg: float,
    max_iters: int,
    ref_score_trans_scale: float,
    ref_score_rot_scale_deg: float,
    show_progress: bool = True,
    desc: str = "estimate seq refs",
) -> Dict[str, np.ndarray]:
    policy = str(ref_policy).lower()
    if policy == "robust_mean":
        return _estimate_seq_reference_extrinsics_from_sample_extrinsics(
            sample_ids=sample_ids,
            sample_extrinsics=sample_extrinsics,
            seq_group_names=seq_group_names,
            huber_trans_delta=float(huber_trans_delta),
            huber_rot_delta_deg=float(huber_rot_delta_deg),
            max_iters=int(max_iters),
            show_progress=show_progress,
            desc=desc,
        )
    if policy == "medoid":
        return _estimate_seq_reference_extrinsics_by_medoid(
            sample_ids=sample_ids,
            sample_extrinsics=sample_extrinsics,
            seq_group_names=seq_group_names,
            trans_scale=float(ref_score_trans_scale),
            rot_scale_deg=float(ref_score_rot_scale_deg),
            show_progress=show_progress,
            desc=f"{desc} (medoid)",
        )
    raise ValueError(f"Unsupported reference policy for sequence reference estimation: {ref_policy}")


def _build_temporal_reliability_from_projected_cameras_by_sequence(
    sample_ids: Sequence[str],
    seq_group_names: Optional[Sequence[str]],
    pred_extr_by_modality: Dict[str, List[Optional[np.ndarray]]],
    seq_rig_extr_by_modality: Dict[str, Dict[str, np.ndarray]],
    target_modality: str,
    modalities: Sequence[str],
    trans_tau: float,
    rot_tau_deg: float,
    show_progress: bool = True,
    workers: int = 1,
) -> Dict[str, Dict[str, float]]:
    mods = [str(m).lower() for m in modalities]
    target_mod = str(target_modality).lower()
    if target_mod not in pred_extr_by_modality:
        raise ValueError(f"Target modality `{target_mod}` missing from predicted extrinsics.")
    num_samples = len(sample_ids)
    stats_by_mod: Dict[str, Dict[str, dict]] = {m: {} for m in mods}

    worker_context = {
        "sample_ids": sample_ids,
        "seq_group_names": seq_group_names,
        "pred_extr_by_modality": pred_extr_by_modality,
        "seq_rig_extr_by_modality": seq_rig_extr_by_modality,
        "target_mod": target_mod,
        "mods": mods,
        "num_samples": num_samples,
    }
    for seq_name, entries in _run_precomp_worker_pool(
        worker_fn=_temporal_reliability_projected_sample_worker,
        context=worker_context,
        num_samples=num_samples,
        workers=workers,
        show_progress=show_progress,
        desc="temporal reliability projected cams",
    ):
        for mod, r, t in entries:
            seq_stats = stats_by_mod[mod].setdefault(
                seq_name,
                {
                    "n": 0,
                    "t_mean": np.zeros(3, dtype=np.float64),
                    "t_m2": np.zeros(3, dtype=np.float64),
                    "r0": None,
                    "a_mean": 0.0,
                    "a_m2": 0.0,
                },
            )
            seq_stats["n"] += 1
            n = int(seq_stats["n"])

            delta_t = t - seq_stats["t_mean"]
            seq_stats["t_mean"] += delta_t / n
            delta_t2 = t - seq_stats["t_mean"]
            seq_stats["t_m2"] += delta_t * delta_t2

            if seq_stats["r0"] is None:
                seq_stats["r0"] = r.copy()
                angle = 0.0
            else:
                angle = _rotation_angle_deg(r, seq_stats["r0"])
            delta_a = angle - seq_stats["a_mean"]
            seq_stats["a_mean"] += delta_a / n
            delta_a2 = angle - seq_stats["a_mean"]
            seq_stats["a_m2"] += delta_a * delta_a2

    out: Dict[str, Dict[str, float]] = {m: {} for m in mods}
    for mod, per_seq in stats_by_mod.items():
        for seq_name, st in per_seq.items():
            n = int(st["n"])
            if n <= 1:
                out[mod][seq_name] = 1.0
                continue
            t_var = st["t_m2"] / max(n - 1, 1)
            t_std = float(np.mean(np.sqrt(np.maximum(t_var, 0.0))))
            a_var = float(st["a_m2"]) / max(n - 1, 1)
            a_std = float(np.sqrt(max(a_var, 0.0)))
            out[mod][seq_name] = _stability_to_score(
                trans_std=t_std,
                rot_std_deg=a_std,
                trans_tau=trans_tau,
                rot_tau_deg=rot_tau_deg,
            )
    return out


def _cross_sensor_scores(
    points_by_modality: Dict[str, np.ndarray],
    target_modality: str,
    tau: float,
) -> Dict[str, float]:
    mods = list(points_by_modality.keys())
    if len(mods) <= 1:
        return {m: 1.0 for m in mods}

    target = str(target_modality).lower()
    if target in points_by_modality:
        anchor = points_by_modality[target]
        tau_safe = max(float(tau), 1e-8)
        scores: Dict[str, float] = {target: 1.0}
        for mod in mods:
            if mod == target:
                continue
            err = float(np.linalg.norm(points_by_modality[mod] - anchor, axis=-1).mean())
            scores[mod] = float(np.clip(np.exp(-err / tau_safe), 1e-8, 1.0))
        return scores

    stacked = np.stack([points_by_modality[m] for m in mods], axis=0)  # [M, J, 3]
    consensus = np.median(stacked, axis=0)
    tau_safe = max(float(tau), 1e-8)
    scores: Dict[str, float] = {}
    for idx, mod in enumerate(mods):
        err = float(np.linalg.norm(stacked[idx] - consensus, axis=-1).mean())
        scores[mod] = float(np.clip(np.exp(-err / tau_safe), 1e-8, 1.0))
    return scores


def _combine_reliability(
    mods: Sequence[str],
    source: str,
    cross_scores: Dict[str, float],
    temporal_scores: Dict[str, float],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    src = str(source).lower()
    for mod in mods:
        s_cross = float(cross_scores.get(mod, 1.0))
        s_temp = float(temporal_scores.get(mod, 1.0))
        if src == "cross_sensor":
            s = s_cross
        elif src == "temporal":
            s = s_temp
        elif src == "hybrid":
            s = s_cross * s_temp
        else:
            s = s_cross
        out[mod] = float(np.clip(s, 1e-8, 1.0))
    return out


def _select_modalities_by_fusion_mode(
    mods: Sequence[str],
    scores: Dict[str, float],
    fusion_mode: str,
    hard_gate_ratio: float,
) -> List[str]:
    if len(mods) == 0:
        return []
    mode = str(fusion_mode).lower()
    if mode != "hard_gate":
        return list(mods)
    max_w = max(float(scores.get(m, 0.0)) for m in mods)
    thresh = max_w * float(hard_gate_ratio)
    kept = [m for m in mods if float(scores.get(m, 0.0)) >= thresh]
    if len(kept) == 0:
        kept = [max(mods, key=lambda m: float(scores.get(m, 0.0)))]
    return kept


def _fuse_points(
    points_by_modality: Dict[str, np.ndarray],
    modalities: Sequence[str],
    scores: Dict[str, float],
    fusion_mode: str,
) -> np.ndarray:
    mode = str(fusion_mode).lower()
    if len(modalities) == 1:
        return points_by_modality[modalities[0]].astype(np.float32)
    if mode == "mean":
        arr = np.stack([points_by_modality[m] for m in modalities], axis=0)
        return arr.mean(axis=0).astype(np.float32)

    w = np.array([max(float(scores.get(m, 0.0)), 1e-8) for m in modalities], dtype=np.float64)
    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= 0:
        w = np.ones_like(w) / float(len(w))
    else:
        w = w / w_sum
    arr = np.stack([points_by_modality[m] for m in modalities], axis=0).astype(np.float64)
    fused = np.sum(arr * w[:, None, None], axis=0)
    return fused.astype(np.float32)


def _project_multisensor_to_target_frame_seqfixed(
    seq_name: str,
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    target_modality: str,
    fusion_modalities: Sequence[str],
    pred_seq_extr_by_modality: Dict[str, Dict[str, np.ndarray]],
    gt_seq_extr_by_modality: Dict[str, Dict[str, np.ndarray]],
    temporal_reliability_by_modality: Dict[str, Dict[str, float]],
    reliability_source: str,
    fusion_mode: str,
    cross_sensor_tau: float,
    hard_gate_ratio: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, Dict[str, float]]:
    target_modality = str(target_modality).lower()
    target_gt_extr = gt_seq_extr_by_modality.get(target_modality, {}).get(seq_name, None)
    target_pred_extr = pred_seq_extr_by_modality.get(target_modality, {}).get(seq_name, None)
    if target_gt_extr is None or target_pred_extr is None:
        return None, None, 0, {}

    h_target_gt = _to_homogeneous(target_gt_extr)
    h_target_pred = _to_homogeneous(target_pred_extr)
    points_by_modality: Dict[str, np.ndarray] = {}
    for mod in fusion_modalities:
        mod = str(mod).lower()
        pred_mod_extr = pred_seq_extr_by_modality.get(mod, {}).get(seq_name, None)
        if pred_mod_extr is None:
            continue

        h_pred_m = _to_homogeneous(pred_mod_extr)
        try:
            h_m_to_target = h_target_pred @ np.linalg.inv(h_pred_m)
        except np.linalg.LinAlgError:
            continue
        h_pred_target_from_m = h_m_to_target @ h_pred_m
        if not np.isfinite(h_pred_target_from_m).all():
            continue
        points_by_modality[mod] = _transform_points(pred_points, h_pred_target_from_m[:3, :4]).astype(np.float32)

    if len(points_by_modality) == 0:
        return None, None, 0, {}

    mods = list(points_by_modality.keys())
    cross_scores = _cross_sensor_scores(
        points_by_modality,
        target_modality=target_modality,
        tau=cross_sensor_tau,
    )
    temp_scores = {
        mod: float(temporal_reliability_by_modality.get(mod, {}).get(seq_name, 1.0))
        for mod in mods
    }
    reliability_scores = _combine_reliability(
        mods=mods,
        source=reliability_source,
        cross_scores=cross_scores,
        temporal_scores=temp_scores,
    )
    kept_modalities = _select_modalities_by_fusion_mode(
        mods=mods,
        scores=reliability_scores,
        fusion_mode=fusion_mode,
        hard_gate_ratio=hard_gate_ratio,
    )
    pred_proj = _fuse_points(
        points_by_modality=points_by_modality,
        modalities=kept_modalities,
        scores=reliability_scores,
        fusion_mode=fusion_mode,
    ).astype(np.float32)

    gt_proj = _transform_points(gt_points, target_gt_extr).astype(np.float32)
    return pred_proj, gt_proj, len(kept_modalities), reliability_scores


def _project_multisensor_to_target_frame_rigfused(
    seq_name: str,
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    target_modality: str,
    fusion_modalities: Sequence[str],
    pred_sample_extr_by_modality: Dict[str, Optional[np.ndarray]],
    pred_target_seq_extr: Dict[str, np.ndarray],
    seq_rig_extr_by_modality: Dict[str, Dict[str, np.ndarray]],
    gt_seq_extr_by_modality: Dict[str, Dict[str, np.ndarray]],
    temporal_reliability_by_modality: Dict[str, Dict[str, float]],
    reliability_source: str,
    fusion_mode: str,
    cross_sensor_tau: float,
    hard_gate_ratio: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, Dict[str, float]]:
    target_modality = str(target_modality).lower()
    target_gt_extr = gt_seq_extr_by_modality.get(target_modality, {}).get(seq_name, None)
    target_pred_ref_extr = pred_target_seq_extr.get(seq_name, None)
    target_pred_cur_extr = pred_sample_extr_by_modality.get(target_modality, None)
    if target_gt_extr is None or target_pred_ref_extr is None or target_pred_cur_extr is None:
        return None, None, 0, {}

    try:
        h_reanchor = _to_homogeneous(target_pred_ref_extr) @ np.linalg.inv(_to_homogeneous(target_pred_cur_extr))
    except np.linalg.LinAlgError:
        return None, None, 0, {}
    if not np.isfinite(h_reanchor).all():
        return None, None, 0, {}

    points_by_modality: Dict[str, np.ndarray] = {}
    for mod in fusion_modalities:
        mod = str(mod).lower()
        mod_extr = pred_sample_extr_by_modality.get(mod, None)
        rig_extr = seq_rig_extr_by_modality.get(mod, {}).get(seq_name, None)
        if mod_extr is None or rig_extr is None:
            continue
        h_proj = _to_homogeneous(rig_extr) @ _to_homogeneous(mod_extr)
        if not np.isfinite(h_proj).all():
            continue
        h_proj = h_reanchor @ h_proj
        if not np.isfinite(h_proj).all():
            continue
        points_by_modality[mod] = _transform_points(pred_points, h_proj[:3, :4]).astype(np.float32)

    if len(points_by_modality) == 0:
        return None, None, 0, {}

    mods = list(points_by_modality.keys())
    cross_scores = _cross_sensor_scores(
        points_by_modality,
        target_modality=target_modality,
        tau=cross_sensor_tau,
    )
    temp_scores = {
        mod: float(temporal_reliability_by_modality.get(mod, {}).get(seq_name, 1.0))
        for mod in mods
    }
    reliability_scores = _combine_reliability(
        mods=mods,
        source=reliability_source,
        cross_scores=cross_scores,
        temporal_scores=temp_scores,
    )
    kept_modalities = _select_modalities_by_fusion_mode(
        mods=mods,
        scores=reliability_scores,
        fusion_mode=fusion_mode,
        hard_gate_ratio=hard_gate_ratio,
    )
    pred_proj = _fuse_points(
        points_by_modality=points_by_modality,
        modalities=kept_modalities,
        scores=reliability_scores,
        fusion_mode=fusion_mode,
    ).astype(np.float32)

    gt_proj = _transform_points(gt_points, target_gt_extr).astype(np.float32)
    return pred_proj, gt_proj, len(kept_modalities), reliability_scores


def _extract_valid_keypoint_pair(
    pred_all,
    gt_all,
    sample_idx: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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


def _process_sample_worker(sample_idx: int):
    if _EVAL_WORKER_CONTEXT is None:
        raise RuntimeError("Evaluation worker context is not initialized.")
    ctx = _EVAL_WORKER_CONTEXT
    mode = str(ctx["mode"])

    pred_kps, gt_kps = _extract_valid_keypoint_pair(ctx["pred_all"], ctx["gt_all"], sample_idx)
    if pred_kps is None or gt_kps is None:
        return ("skip", sample_idx, None, None, None, None, None)

    sample_ids = ctx["sample_ids"]
    seq_group_names = ctx.get("seq_group_names", None)
    sid = sample_ids[sample_idx] if sample_idx < len(sample_ids) else None
    if seq_group_names is not None and sample_idx < len(seq_group_names):
        seq_name = str(seq_group_names[sample_idx])
    else:
        seq_name = _seq_name_from_sample_id(sid)
    frame_idx = _frame_index_from_sample_id(sid)

    if mode == "seq_lidar_ref":
        pred_extr = ctx["pred_seq_extr"].get(seq_name, None)
        gt_extr = ctx["gt_seq_extr"].get(seq_name, None)
        if pred_extr is None or gt_extr is None:
            return ("no_seq_ref", sample_idx, None, None, None, None, None)
        pred_proj = _transform_points(pred_kps, pred_extr)
        gt_proj = _transform_points(gt_kps, gt_extr)
        return ("ok", sample_idx, seq_name, frame_idx, pred_proj, gt_proj, None)

    if mode == "multi_sensor_seqfixed":
        pred_proj, gt_proj, used_mods, reliability_scores = _project_multisensor_to_target_frame_seqfixed(
            seq_name=seq_name,
            pred_points=pred_kps,
            gt_points=gt_kps,
            target_modality=ctx["target_modality"],
            fusion_modalities=ctx["active_fusion_modalities"],
            pred_seq_extr_by_modality=ctx["pred_seq_extr_by_modality"],
            gt_seq_extr_by_modality=ctx["gt_seq_extr_by_modality"],
            temporal_reliability_by_modality=ctx["temporal_reliability_by_modality"],
            reliability_source=ctx["reliability_source"],
            fusion_mode=ctx["fusion_mode"],
            cross_sensor_tau=ctx["cross_sensor_tau"],
            hard_gate_ratio=ctx["hard_gate_ratio"],
        )
        if pred_proj is None or gt_proj is None:
            return ("no_projection", sample_idx, None, None, None, None, None)
        return (
            "ok",
            sample_idx,
            seq_name,
            frame_idx,
            pred_proj,
            gt_proj,
            (int(used_mods), reliability_scores),
        )

    if mode == "multi_sensor_rigfused":
        num_samples = int(ctx["num_samples"])
        pred_extr_by_modality = ctx["pred_extr_by_modality"]
        pred_sample_extr_by_modality = {
            mod: pred_extr_by_modality.get(mod, [None] * num_samples)[sample_idx]
            for mod in ctx["active_fusion_modalities"]
        }
        pred_proj, gt_proj, used_mods, reliability_scores = _project_multisensor_to_target_frame_rigfused(
            seq_name=seq_name,
            pred_points=pred_kps,
            gt_points=gt_kps,
            target_modality=ctx["target_modality"],
            fusion_modalities=ctx["active_fusion_modalities"],
            pred_sample_extr_by_modality=pred_sample_extr_by_modality,
            pred_target_seq_extr=ctx["pred_target_seq_extr"],
            seq_rig_extr_by_modality=ctx["seq_rig_extr_by_modality"],
            gt_seq_extr_by_modality=ctx["gt_seq_extr_by_modality"],
            temporal_reliability_by_modality=ctx["temporal_reliability_by_modality"],
            reliability_source=ctx["reliability_source"],
            fusion_mode=ctx["fusion_mode"],
            cross_sensor_tau=ctx["cross_sensor_tau"],
            hard_gate_ratio=ctx["hard_gate_ratio"],
        )
        if pred_proj is None or gt_proj is None:
            return ("no_projection", sample_idx, None, None, None, None, None)
        return (
            "ok",
            sample_idx,
            seq_name,
            frame_idx,
            pred_proj,
            gt_proj,
            (int(used_mods), reliability_scores),
        )

    raise ValueError(f"Unknown worker mode `{mode}`.")


def _run_sample_worker_pool(
    context: Dict[str, Any],
    num_samples: int,
    workers: int,
    show_progress: bool,
    desc: str = "transform keypoints",
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


def _format_mm(x_meters: float) -> str:
    return f"{x_meters:.6f} m ({x_meters * 1000.0:.3f} mm)"


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
                # Causal smoothing: use current + past frames only, no future leakage.
                l = max(0, p - window + 1)
                r = p + 1
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
    parser.add_argument("--pred-cameras-key", type=str, default="pred_cameras_stream")
    parser.add_argument("--gt-cameras-key", type=str, default="gt_cameras_stream")
    parser.add_argument("--pose-encoding-type", type=str, default="absT_quaR_FoV")
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
    parser.add_argument(
        "--sensor-index-by-modality",
        type=str,
        default="",
        help="Comma-separated `modality:sensor_idx` (e.g., `lidar:0,rgb:1`) for stream camera keys.",
    )
    parser.add_argument(
        "--projection-mode",
        type=str,
        default="seq_lidar_ref",
        choices=["seq_lidar_ref", "multi_sensor", "multi_sensor_rig_fused"],
        help=(
            "Projection strategy. "
            "`seq_lidar_ref`: keep old fixed per-sequence LiDAR-camera projection. "
            "`multi_sensor`: fuse fixed per-sequence cameras from multiple modalities into a target frame. "
            "`multi_sensor_rig_fused`: estimate fixed sensor-to-target transforms per sequence from predicted "
            "cameras, then project each sensor stream via those transforms before fusion."
        ),
    )
    parser.add_argument(
        "--target-modality",
        type=str,
        default="lidar",
        help=(
            "Target sensor frame for `multi_sensor` and `multi_sensor_rig_fused` "
            "projection modes (e.g., lidar, rgb)."
        ),
    )
    parser.add_argument(
        "--fusion-modalities",
        type=str,
        default="rgb,lidar",
        help=(
            "Comma-separated modalities to fuse in `multi_sensor` and "
            "`multi_sensor_rig_fused` modes (e.g., rgb,lidar)."
        ),
    )
    parser.add_argument(
        "--fusion-mode",
        type=str,
        default="mean",
        choices=["mean", "weighted", "hard_gate"],
        help="How to fuse modalities in multi-sensor projection modes.",
    )
    parser.add_argument(
        "--reliability-source",
        type=str,
        default="cross_sensor",
        choices=["cross_sensor", "temporal", "hybrid"],
        help="Reliability scoring source for robust multi-sensor fusion.",
    )
    parser.add_argument(
        "--cross-sensor-tau",
        type=float,
        default=0.05,
        help="Meters. Smaller values penalize cross-sensor disagreement more strongly.",
    )
    parser.add_argument(
        "--temporal-trans-tau",
        type=float,
        default=0.05,
        help="Meters. Translation stability scale for temporal reliability.",
    )
    parser.add_argument(
        "--temporal-rot-tau-deg",
        type=float,
        default=5.0,
        help="Degrees. Rotation stability scale for temporal reliability.",
    )
    parser.add_argument(
        "--hard-gate-ratio",
        type=float,
        default=0.8,
        help="Keep modalities with reliability >= hard_gate_ratio * best_reliability.",
    )
    parser.add_argument(
        "--rig-mean-huber-trans-delta",
        type=float,
        default=0.10,
        help=(
            "Meters. Huber delta for robust sequence-wise sensor-to-target translation averaging "
            "in `multi_sensor_rig_fused`."
        ),
    )
    parser.add_argument(
        "--rig-mean-huber-rot-delta-deg",
        type=float,
        default=10.0,
        help=(
            "Degrees. Huber delta for robust sequence-wise sensor-to-target rotation averaging "
            "in `multi_sensor_rig_fused`."
        ),
    )
    parser.add_argument(
        "--rig-mean-max-iters",
        type=int,
        default=20,
        help="Max IRLS iterations for robust rig transform averaging in `multi_sensor_rig_fused`.",
    )
    parser.add_argument(
        "--seq-ref-policy",
        type=str,
        default="first",
        choices=["first", "best_sensor", "robust_mean", "medoid"],
        help=(
            "Per-sequence reference-camera policy. "
            "`first` keeps legacy first-valid behavior. "
            "`best_sensor` tries each sensor tag in a sequence and selects the most stable reference. "
            "`robust_mean` uses robust sequence-wise mean camera extrinsics (anchor-free). "
            "`medoid` picks the real frame camera minimizing median normalized "
            "rotation+translation distance to other frames in the sequence."
        ),
    )
    parser.add_argument(
        "--seq-ref-score-trans-scale",
        type=float,
        default=0.10,
        help=(
            "Meters. Translation scale for `best_sensor` and `medoid` reference scoring. "
            "Lower values penalize translation inconsistency more."
        ),
    )
    parser.add_argument(
        "--seq-ref-score-rot-scale-deg",
        type=float,
        default=10.0,
        help=(
            "Degrees. Rotation scale for `best_sensor` and `medoid` reference scoring. "
            "Lower values penalize rotation inconsistency more."
        ),
    )
    parser.add_argument("--pelvis-index", type=int, default=0, help="Pelvis joint index for centering.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for per-sample projection/transformation. 1 disables multiprocessing.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help=(
            "Temporal smoothing window per sequence. "
            "For --smooth-mode centered, window must be odd. 1 disables smoothing."
        ),
    )
    parser.add_argument(
        "--smooth-on",
        type=str,
        default="pred",
        choices=["pred", "gt", "both"],
        help="Which keypoints to smooth before metric computation.",
    )
    parser.add_argument(
        "--smooth-mode",
        type=str,
        default="centered",
        choices=["centered", "causal"],
        help="Smoothing mode: centered uses past+future, causal uses only past+current.",
    )
    args = parser.parse_args()
    show_progress = not args.no_progress
    if args.workers <= 0:
        raise ValueError(f"--workers must be >= 1, got {args.workers}")
    args.hard_gate_ratio = float(np.clip(float(args.hard_gate_ratio), 0.0, 1.0))
    if args.seq_ref_score_trans_scale <= 0.0:
        raise ValueError(
            f"--seq-ref-score-trans-scale must be > 0, got {args.seq_ref_score_trans_scale}"
        )
    if args.seq_ref_score_rot_scale_deg <= 0.0:
        raise ValueError(
            f"--seq-ref-score-rot-scale-deg must be > 0, got {args.seq_ref_score_rot_scale_deg}"
        )

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
    seq_group_names = _build_seq_group_names(data, sample_ids)

    pred_new = []
    gt_new = []
    kept_seq_names = []
    kept_frame_indices = []
    kept = 0
    projection_mode = str(args.projection_mode).lower()
    ref_policy = str(args.seq_ref_policy).lower()
    target_modality = str(args.target_modality).lower()
    fusion_modalities = _parse_modalities_list(args.fusion_modalities)
    modality_sensor_indices = _parse_sensor_index_map(args.sensor_index_by_modality)
    if "lidar" not in modality_sensor_indices:
        modality_sensor_indices["lidar"] = 0
    if "rgb" not in modality_sensor_indices:
        modality_sensor_indices["rgb"] = 0
    modality_fallback_indices: Dict[str, Optional[int]] = {
        "lidar": args.lidar_modality_index,
        "rgb": args.rgb_modality_index,
    }
    projection_info: List[str] = []
    projection_info.append(f"[INFO] Camera keys: pred={pred_camera_key}, gt={gt_camera_key}")
    projection_info.append(f"[INFO] Worker processes: {int(args.workers)}")
    projection_info.append(f"[INFO] Sequence ref policy: {str(args.seq_ref_policy).lower()}")

    if projection_mode == "seq_lidar_ref":
        pred_seq_meta: Dict[str, Tuple[int, int]] = {}
        gt_seq_meta: Dict[str, Tuple[int, int]] = {}
        chosen_sensor_by_seq: Dict[str, str] = {}
        if ref_policy in {"robust_mean", "medoid"}:
            pred_extr_by_modality = _build_pred_extrinsics_by_sample_for_modalities(
                data=data,
                camera_key=pred_camera_key,
                sample_ids=sample_ids,
                num_samples=num_samples,
                modalities=["lidar"],
                modality_fallback_indices=modality_fallback_indices,
                modality_sensor_indices=modality_sensor_indices,
                pose_encoding_type=args.pose_encoding_type,
                show_progress=show_progress,
                workers=int(args.workers),
            )
            gt_extr_by_modality = _build_pred_extrinsics_by_sample_for_modalities(
                data=data,
                camera_key=gt_camera_key,
                sample_ids=sample_ids,
                num_samples=num_samples,
                modalities=["lidar"],
                modality_fallback_indices=modality_fallback_indices,
                modality_sensor_indices=modality_sensor_indices,
                pose_encoding_type=args.pose_encoding_type,
                show_progress=show_progress,
                workers=int(args.workers),
            )
            pred_seq_extr = _estimate_seq_reference_extrinsics_with_policy(
                ref_policy=ref_policy,
                sample_ids=sample_ids,
                sample_extrinsics=pred_extr_by_modality.get("lidar", []),
                seq_group_names=seq_group_names,
                huber_trans_delta=float(args.rig_mean_huber_trans_delta),
                huber_rot_delta_deg=float(args.rig_mean_huber_rot_delta_deg),
                max_iters=int(args.rig_mean_max_iters),
                ref_score_trans_scale=float(args.seq_ref_score_trans_scale),
                ref_score_rot_scale_deg=float(args.seq_ref_score_rot_scale_deg),
                show_progress=show_progress,
                desc=f"estimate seq refs {pred_camera_key}:lidar",
            )
            gt_seq_extr = _estimate_seq_reference_extrinsics_with_policy(
                ref_policy=ref_policy,
                sample_ids=sample_ids,
                sample_extrinsics=gt_extr_by_modality.get("lidar", []),
                seq_group_names=seq_group_names,
                huber_trans_delta=float(args.rig_mean_huber_trans_delta),
                huber_rot_delta_deg=float(args.rig_mean_huber_rot_delta_deg),
                max_iters=int(args.rig_mean_max_iters),
                ref_score_trans_scale=float(args.seq_ref_score_trans_scale),
                ref_score_rot_scale_deg=float(args.seq_ref_score_rot_scale_deg),
                show_progress=show_progress,
                desc=f"estimate seq refs {gt_camera_key}:lidar",
            )
        else:
            pred_seq_cam_enc, pred_seq_meta, gt_seq_cam_enc, gt_seq_meta, chosen_sensor_by_seq = (
                _build_paired_sequence_reference_cameras_for_modality(
                    data=data,
                    pred_camera_key=pred_camera_key,
                    gt_camera_key=gt_camera_key,
                    sample_ids=sample_ids,
                    num_samples=num_samples,
                    modality="lidar",
                    fallback_modality_idx=args.lidar_modality_index,
                    sensor_idx=modality_sensor_indices.get("lidar", 0),
                    ref_policy=ref_policy,
                    pose_encoding_type=args.pose_encoding_type,
                    ref_score_trans_scale=float(args.seq_ref_score_trans_scale),
                    ref_score_rot_scale_deg=float(args.seq_ref_score_rot_scale_deg),
                    show_progress=show_progress,
                )
            )
            pred_seq_extr = {
                seq: _pose_encoding_to_extrinsic(cam_enc, args.pose_encoding_type)
                for seq, cam_enc in pred_seq_cam_enc.items()
            }
            gt_seq_extr = {
                seq: _pose_encoding_to_extrinsic(cam_enc, args.pose_encoding_type)
                for seq, cam_enc in gt_seq_cam_enc.items()
            }

        dropped_no_seq_ref = 0
        worker_context = {
            "mode": "seq_lidar_ref",
            "pred_all": pred_all,
            "gt_all": gt_all,
            "sample_ids": sample_ids,
            "seq_group_names": seq_group_names,
            "pred_seq_extr": pred_seq_extr,
            "gt_seq_extr": gt_seq_extr,
        }
        for status, _i, seq_name, frame_idx, pred_proj, gt_proj, _extra in _run_sample_worker_pool(
            context=worker_context,
            num_samples=num_samples,
            workers=int(args.workers),
            show_progress=show_progress,
            desc="transform keypoints",
        ):
            if status == "no_seq_ref":
                dropped_no_seq_ref += 1
                continue
            if status != "ok":
                continue
            pred_new.append(pred_proj)
            gt_new.append(gt_proj)
            kept_seq_names.append(seq_name)
            kept_frame_indices.append(frame_idx)
            kept += 1

        projection_info.append(f"[INFO] Sequence reference cameras (pred): {len(pred_seq_extr)}")
        projection_info.append(f"[INFO] Sequence reference cameras (gt):   {len(gt_seq_extr)}")
        if ref_policy == "best_sensor":
            projection_info.append(
                f"[INFO] Best-sensor refs selected: {len(chosen_sensor_by_seq)} sequences"
            )
        if ref_policy == "robust_mean":
            projection_info.append("[INFO] Anchor-free LiDAR refs estimated by robust sequence means.")
        if ref_policy == "medoid":
            projection_info.append(
                "[INFO] Anchor-free LiDAR refs estimated by sequence medoid cameras."
            )
        example_seq = sorted(set(pred_seq_meta.keys()) & set(gt_seq_meta.keys()))
        if len(example_seq) > 0:
            seq0 = example_seq[0]
            p_idx, p_m = pred_seq_meta[seq0]
            g_idx, g_m = gt_seq_meta[seq0]
            projection_info.append(
                f"[INFO] Example seq `{seq0}` refs: "
                f"pred(sample_idx={p_idx}, camera_idx={p_m}), "
                f"gt(sample_idx={g_idx}, camera_idx={g_m})"
            )
        if dropped_no_seq_ref > 0:
            projection_info.append(f"[INFO] Dropped samples without per-sequence refs: {dropped_no_seq_ref}")
    else:
        if len(fusion_modalities) == 0:
            raise ValueError("`--fusion-modalities` must contain at least one modality for multi-sensor modes.")

        required_modalities = sorted(set(fusion_modalities + [target_modality]))
        reliability_source = str(args.reliability_source).lower()
        fusion_mode = str(args.fusion_mode).lower()

        if projection_mode == "multi_sensor":
            pred_seq_extr_by_modality: Dict[str, Dict[str, np.ndarray]] = {}
            gt_seq_extr_by_modality: Dict[str, Dict[str, np.ndarray]] = {}
            seq_meta_by_modality: Dict[str, Tuple[Dict[str, Tuple[int, int]], Dict[str, Tuple[int, int]]]] = {}
            if ref_policy in {"robust_mean", "medoid"}:
                pred_extr_by_modality = _build_pred_extrinsics_by_sample_for_modalities(
                    data=data,
                    camera_key=pred_camera_key,
                    sample_ids=sample_ids,
                    num_samples=num_samples,
                    modalities=required_modalities,
                    modality_fallback_indices=modality_fallback_indices,
                    modality_sensor_indices=modality_sensor_indices,
                    pose_encoding_type=args.pose_encoding_type,
                    show_progress=show_progress,
                    workers=int(args.workers),
                )
                for mod in required_modalities:
                    try:
                        pred_seq_extr_by_modality[mod] = _estimate_seq_reference_extrinsics_with_policy(
                            ref_policy=ref_policy,
                            sample_ids=sample_ids,
                            sample_extrinsics=pred_extr_by_modality.get(mod, []),
                            seq_group_names=seq_group_names,
                            huber_trans_delta=float(args.rig_mean_huber_trans_delta),
                            huber_rot_delta_deg=float(args.rig_mean_huber_rot_delta_deg),
                            max_iters=int(args.rig_mean_max_iters),
                            ref_score_trans_scale=float(args.seq_ref_score_trans_scale),
                            ref_score_rot_scale_deg=float(args.seq_ref_score_rot_scale_deg),
                            show_progress=show_progress,
                            desc=f"estimate seq refs {pred_camera_key}:{mod}",
                        )
                    except ValueError:
                        pass
                    seq_meta_by_modality[mod] = ({}, {})

                gt_extr_by_modality = _build_pred_extrinsics_by_sample_for_modalities(
                    data=data,
                    camera_key=gt_camera_key,
                    sample_ids=sample_ids,
                    num_samples=num_samples,
                    modalities=[target_modality],
                    modality_fallback_indices=modality_fallback_indices,
                    modality_sensor_indices=modality_sensor_indices,
                    pose_encoding_type=args.pose_encoding_type,
                    show_progress=show_progress,
                    workers=int(args.workers),
                )
                gt_seq_extr_by_modality[target_modality] = _estimate_seq_reference_extrinsics_with_policy(
                    ref_policy=ref_policy,
                    sample_ids=sample_ids,
                    sample_extrinsics=gt_extr_by_modality.get(target_modality, []),
                    seq_group_names=seq_group_names,
                    huber_trans_delta=float(args.rig_mean_huber_trans_delta),
                    huber_rot_delta_deg=float(args.rig_mean_huber_rot_delta_deg),
                    max_iters=int(args.rig_mean_max_iters),
                    ref_score_trans_scale=float(args.seq_ref_score_trans_scale),
                    ref_score_rot_scale_deg=float(args.seq_ref_score_rot_scale_deg),
                    show_progress=show_progress,
                    desc=f"estimate seq refs {gt_camera_key}:{target_modality}",
                )
            else:
                for mod in required_modalities:
                    fb_idx = modality_fallback_indices.get(mod, None)
                    pred_seq_cam_enc = None
                    pred_seq_meta = {}
                    gt_seq_cam_enc = None
                    gt_seq_meta = {}
                    if mod == target_modality:
                        try:
                            (
                                pred_seq_cam_enc,
                                pred_seq_meta,
                                gt_seq_cam_enc,
                                gt_seq_meta,
                                _chosen_sensor_by_seq,
                            ) = _build_paired_sequence_reference_cameras_for_modality(
                                data=data,
                                pred_camera_key=pred_camera_key,
                                gt_camera_key=gt_camera_key,
                                sample_ids=sample_ids,
                                num_samples=num_samples,
                                modality=mod,
                                fallback_modality_idx=fb_idx,
                                sensor_idx=modality_sensor_indices.get(mod, 0),
                                ref_policy=ref_policy,
                                pose_encoding_type=args.pose_encoding_type,
                                ref_score_trans_scale=float(args.seq_ref_score_trans_scale),
                                ref_score_rot_scale_deg=float(args.seq_ref_score_rot_scale_deg),
                                show_progress=show_progress,
                            )
                        except ValueError:
                            gt_seq_cam_enc = None
                            pred_seq_cam_enc = None
                    else:
                        try:
                            (
                                pred_seq_cam_enc,
                                pred_seq_meta,
                                _chosen_sensor_by_seq,
                            ) = _build_sequence_reference_cameras_for_modality_with_policy(
                                data=data,
                                camera_key=pred_camera_key,
                                sample_ids=sample_ids,
                                num_samples=num_samples,
                                modality=mod,
                                fallback_modality_idx=fb_idx,
                                sensor_idx=modality_sensor_indices.get(mod, 0),
                                ref_policy=ref_policy,
                                pose_encoding_type=args.pose_encoding_type,
                                ref_score_trans_scale=float(args.seq_ref_score_trans_scale),
                                ref_score_rot_scale_deg=float(args.seq_ref_score_rot_scale_deg),
                                preferred_sensor_by_seq=None,
                                show_progress=show_progress,
                            )
                        except ValueError:
                            pred_seq_cam_enc = None

                    if pred_seq_cam_enc is not None:
                        pred_seq_extr_by_modality[mod] = {
                            seq: _pose_encoding_to_extrinsic(cam_enc, args.pose_encoding_type)
                            for seq, cam_enc in pred_seq_cam_enc.items()
                        }
                    if gt_seq_cam_enc is not None:
                        gt_seq_extr_by_modality[mod] = {
                            seq: _pose_encoding_to_extrinsic(cam_enc, args.pose_encoding_type)
                            for seq, cam_enc in gt_seq_cam_enc.items()
                        }
                    seq_meta_by_modality[mod] = (pred_seq_meta, gt_seq_meta)

            if target_modality not in gt_seq_extr_by_modality:
                raise ValueError(
                    f"`multi_sensor` requires GT sequence refs for target modality `{target_modality}` "
                    f"from key `{gt_camera_key}`."
                )
            if target_modality not in pred_seq_extr_by_modality:
                raise ValueError(
                    f"`multi_sensor` requires predicted sequence refs for target modality "
                    f"`{target_modality}` from key `{pred_camera_key}`."
                )

            active_fusion_modalities = [
                mod
                for mod in fusion_modalities
                if mod in pred_seq_extr_by_modality
            ]
            if len(active_fusion_modalities) == 0:
                raise ValueError(
                    "No fusion modalities have predicted sequence references. "
                    f"Requested={fusion_modalities}"
                )

            temporal_reliability_by_modality: Dict[str, Dict[str, float]] = {}
            if reliability_source in {"temporal", "hybrid"}:
                temporal_reliability_by_modality = _build_temporal_reliability_by_sequence(
                    data=data,
                    camera_key=pred_camera_key,
                    sample_ids=sample_ids,
                    num_samples=num_samples,
                    modalities=active_fusion_modalities,
                    target_modality=target_modality,
                    modality_fallback_indices=modality_fallback_indices,
                    modality_sensor_indices=modality_sensor_indices,
                    pose_encoding_type=args.pose_encoding_type,
                    trans_tau=float(args.temporal_trans_tau),
                    rot_tau_deg=float(args.temporal_rot_tau_deg),
                    show_progress=show_progress,
                    workers=int(args.workers),
                )

            projection_info.append(
                f"[INFO] Multi-sensor projection (seq-fixed): target={target_modality}, "
                f"requested_fusion={fusion_modalities}, active_fusion={active_fusion_modalities}"
            )
            projection_info.append(
                f"[INFO] Fusion mode: {fusion_mode}, reliability: {reliability_source}"
            )
            if ref_policy == "robust_mean":
                projection_info.append(
                    "[INFO] Anchor-free sequence refs estimated by robust sequence means."
                )
            if ref_policy == "medoid":
                projection_info.append(
                    "[INFO] Anchor-free sequence refs estimated by sequence medoid cameras."
                )
            if fusion_mode == "hard_gate":
                projection_info.append(f"[INFO] Hard gate ratio: {args.hard_gate_ratio:.3f}")
            for mod in required_modalities:
                pred_count = len(pred_seq_extr_by_modality.get(mod, {}))
                gt_count = len(gt_seq_extr_by_modality.get(mod, {}))
                projection_info.append(
                    f"[INFO] Sequence refs `{mod}`: pred={pred_count}, gt={gt_count}"
                )
                pred_meta, gt_meta = seq_meta_by_modality.get(mod, ({}, {}))
                example_seq = sorted(set(pred_meta.keys()) & set(gt_meta.keys()))
                if len(example_seq) > 0:
                    seq0 = example_seq[0]
                    p_idx, p_m = pred_meta[seq0]
                    g_idx, g_m = gt_meta[seq0]
                    projection_info.append(
                        f"[INFO] Example seq `{seq0}` refs `{mod}`: "
                        f"pred(sample_idx={p_idx}, camera_idx={p_m}), "
                        f"gt(sample_idx={g_idx}, camera_idx={g_m})"
                    )
            if reliability_source in {"temporal", "hybrid"}:
                for mod in active_fusion_modalities:
                    vals = list(temporal_reliability_by_modality.get(mod, {}).values())
                    if len(vals) == 0:
                        continue
                    projection_info.append(
                        f"[INFO] Temporal reliability `{mod}`: mean={np.mean(vals):.4f}, "
                        f"min={np.min(vals):.4f}, max={np.max(vals):.4f}"
                    )

            used_modality_counts = []
            rel_values_by_modality: Dict[str, List[float]] = {mod: [] for mod in active_fusion_modalities}
            dropped_no_camera_projection = 0
            worker_context = {
                "mode": "multi_sensor_seqfixed",
                "pred_all": pred_all,
                "gt_all": gt_all,
                "sample_ids": sample_ids,
                "seq_group_names": seq_group_names,
                "target_modality": target_modality,
                "active_fusion_modalities": active_fusion_modalities,
                "pred_seq_extr_by_modality": pred_seq_extr_by_modality,
                "gt_seq_extr_by_modality": gt_seq_extr_by_modality,
                "temporal_reliability_by_modality": temporal_reliability_by_modality,
                "reliability_source": reliability_source,
                "fusion_mode": fusion_mode,
                "cross_sensor_tau": float(args.cross_sensor_tau),
                "hard_gate_ratio": float(args.hard_gate_ratio),
            }
            for status, _i, seq_name, frame_idx, pred_proj, gt_proj, extra in _run_sample_worker_pool(
                context=worker_context,
                num_samples=num_samples,
                workers=int(args.workers),
                show_progress=show_progress,
                desc="transform keypoints",
            ):
                if status == "no_projection":
                    dropped_no_camera_projection += 1
                    continue
                if status != "ok":
                    continue

                used_mods, reliability_scores = extra
                pred_new.append(pred_proj)
                gt_new.append(gt_proj)
                used_modality_counts.append(int(used_mods))
                for mod, val in reliability_scores.items():
                    if mod in rel_values_by_modality:
                        rel_values_by_modality[mod].append(float(val))
                kept_seq_names.append(seq_name)
                kept_frame_indices.append(frame_idx)
                kept += 1

            if len(used_modality_counts) > 0:
                projection_info.append(
                    f"[INFO] Modalities used per sample: mean={np.mean(used_modality_counts):.2f}, "
                    f"min={np.min(used_modality_counts)}, max={np.max(used_modality_counts)}"
                )
            for mod in active_fusion_modalities:
                vals = rel_values_by_modality.get(mod, [])
                if len(vals) == 0:
                    continue
                projection_info.append(
                    f"[INFO] Reliability `{mod}`: mean={np.mean(vals):.4f}, "
                    f"min={np.min(vals):.4f}, max={np.max(vals):.4f}"
                )
            if dropped_no_camera_projection > 0:
                projection_info.append(
                    f"[INFO] Dropped samples without valid multi-sensor camera projection: "
                    f"{dropped_no_camera_projection}"
                )
        else:
            fb_idx = modality_fallback_indices.get(target_modality, None)
            pred_extr_by_modality = _build_pred_extrinsics_by_sample_for_modalities(
                data=data,
                camera_key=pred_camera_key,
                sample_ids=sample_ids,
                num_samples=num_samples,
                modalities=required_modalities,
                modality_fallback_indices=modality_fallback_indices,
                modality_sensor_indices=modality_sensor_indices,
                pose_encoding_type=args.pose_encoding_type,
                show_progress=show_progress,
                workers=int(args.workers),
            )
            pred_target_seq_meta: Dict[str, Tuple[int, int]] = {}
            gt_seq_meta: Dict[str, Tuple[int, int]] = {}
            chosen_sensor_by_seq: Dict[str, str] = {}
            if ref_policy in {"robust_mean", "medoid"}:
                pred_target_seq_extr = _estimate_seq_reference_extrinsics_with_policy(
                    ref_policy=ref_policy,
                    sample_ids=sample_ids,
                    sample_extrinsics=pred_extr_by_modality.get(target_modality, []),
                    seq_group_names=seq_group_names,
                    huber_trans_delta=float(args.rig_mean_huber_trans_delta),
                    huber_rot_delta_deg=float(args.rig_mean_huber_rot_delta_deg),
                    max_iters=int(args.rig_mean_max_iters),
                    ref_score_trans_scale=float(args.seq_ref_score_trans_scale),
                    ref_score_rot_scale_deg=float(args.seq_ref_score_rot_scale_deg),
                    show_progress=show_progress,
                    desc=f"estimate target refs {pred_camera_key}:{target_modality}",
                )
                gt_extr_by_modality = _build_pred_extrinsics_by_sample_for_modalities(
                    data=data,
                    camera_key=gt_camera_key,
                    sample_ids=sample_ids,
                    num_samples=num_samples,
                    modalities=[target_modality],
                    modality_fallback_indices=modality_fallback_indices,
                    modality_sensor_indices=modality_sensor_indices,
                    pose_encoding_type=args.pose_encoding_type,
                    show_progress=show_progress,
                    workers=int(args.workers),
                )
                gt_target_seq_extr = _estimate_seq_reference_extrinsics_with_policy(
                    ref_policy=ref_policy,
                    sample_ids=sample_ids,
                    sample_extrinsics=gt_extr_by_modality.get(target_modality, []),
                    seq_group_names=seq_group_names,
                    huber_trans_delta=float(args.rig_mean_huber_trans_delta),
                    huber_rot_delta_deg=float(args.rig_mean_huber_rot_delta_deg),
                    max_iters=int(args.rig_mean_max_iters),
                    ref_score_trans_scale=float(args.seq_ref_score_trans_scale),
                    ref_score_rot_scale_deg=float(args.seq_ref_score_rot_scale_deg),
                    show_progress=show_progress,
                    desc=f"estimate target refs {gt_camera_key}:{target_modality}",
                )
                gt_seq_extr_by_modality: Dict[str, Dict[str, np.ndarray]] = {
                    target_modality: gt_target_seq_extr
                }
            else:
                (
                    pred_target_seq_cam_enc,
                    pred_target_seq_meta,
                    gt_seq_cam_enc,
                    gt_seq_meta,
                    chosen_sensor_by_seq,
                ) = _build_paired_sequence_reference_cameras_for_modality(
                    data=data,
                    pred_camera_key=pred_camera_key,
                    gt_camera_key=gt_camera_key,
                    sample_ids=sample_ids,
                    num_samples=num_samples,
                    modality=target_modality,
                    fallback_modality_idx=fb_idx,
                    sensor_idx=modality_sensor_indices.get(target_modality, 0),
                    ref_policy=ref_policy,
                    pose_encoding_type=args.pose_encoding_type,
                    ref_score_trans_scale=float(args.seq_ref_score_trans_scale),
                    ref_score_rot_scale_deg=float(args.seq_ref_score_rot_scale_deg),
                    show_progress=show_progress,
                )
                pred_target_seq_extr = {
                    seq: _pose_encoding_to_extrinsic(cam_enc, args.pose_encoding_type)
                    for seq, cam_enc in pred_target_seq_cam_enc.items()
                }
                gt_seq_extr_by_modality = {
                    target_modality: {
                        seq: _pose_encoding_to_extrinsic(cam_enc, args.pose_encoding_type)
                        for seq, cam_enc in gt_seq_cam_enc.items()
                    }
                }

            seq_rig_extr_by_modality = _estimate_seq_rig_transforms_from_predicted_cameras(
                sample_ids=sample_ids,
                seq_group_names=seq_group_names,
                pred_extr_by_modality=pred_extr_by_modality,
                target_modality=target_modality,
                modalities=required_modalities,
                huber_trans_delta=float(args.rig_mean_huber_trans_delta),
                huber_rot_delta_deg=float(args.rig_mean_huber_rot_delta_deg),
                max_iters=int(args.rig_mean_max_iters),
                show_progress=show_progress,
                workers=int(args.workers),
            )

            active_fusion_modalities = [
                mod
                for mod in fusion_modalities
                if len(seq_rig_extr_by_modality.get(mod, {})) > 0
            ]
            if len(active_fusion_modalities) == 0:
                raise ValueError(
                    "No fusion modalities have valid sequence rig transforms. "
                    f"Requested={fusion_modalities}"
                )

            temporal_reliability_by_modality: Dict[str, Dict[str, float]] = {}
            if reliability_source in {"temporal", "hybrid"}:
                temporal_reliability_by_modality = _build_temporal_reliability_from_projected_cameras_by_sequence(
                    sample_ids=sample_ids,
                    seq_group_names=seq_group_names,
                    pred_extr_by_modality=pred_extr_by_modality,
                    seq_rig_extr_by_modality=seq_rig_extr_by_modality,
                    target_modality=target_modality,
                    modalities=active_fusion_modalities,
                    trans_tau=float(args.temporal_trans_tau),
                    rot_tau_deg=float(args.temporal_rot_tau_deg),
                    show_progress=show_progress,
                    workers=int(args.workers),
                )

            projection_info.append(
                f"[INFO] Multi-sensor projection (rig-fused): target={target_modality}, "
                f"requested_fusion={fusion_modalities}, active_fusion={active_fusion_modalities}"
            )
            projection_info.append(
                f"[INFO] Fusion mode: {fusion_mode}, reliability: {reliability_source}"
            )
            projection_info.append(
                f"[INFO] Rig mean IRLS: trans_delta={float(args.rig_mean_huber_trans_delta):.4f} m, "
                f"rot_delta={float(args.rig_mean_huber_rot_delta_deg):.2f} deg, "
                f"max_iters={int(args.rig_mean_max_iters)}"
            )
            if fusion_mode == "hard_gate":
                projection_info.append(f"[INFO] Hard gate ratio: {args.hard_gate_ratio:.3f}")
            for mod in required_modalities:
                pred_valid = int(sum(1 for e in pred_extr_by_modality.get(mod, []) if e is not None))
                rig_count = len(seq_rig_extr_by_modality.get(mod, {}))
                gt_count = len(gt_seq_extr_by_modality.get(mod, {}))
                projection_info.append(
                    f"[INFO] Cameras `{mod}`: pred_valid_frames={pred_valid}, "
                    f"seq_rig_refs={rig_count}, gt_seq_refs={gt_count}"
                )
            projection_info.append(
                f"[INFO] Target fixed refs `{target_modality}`: pred_seq_refs={len(pred_target_seq_extr)}, "
                f"gt_seq_refs={len(gt_seq_extr_by_modality.get(target_modality, {}))}"
            )
            if ref_policy == "best_sensor":
                projection_info.append(
                    f"[INFO] Best-sensor refs selected `{target_modality}`: "
                    f"{len(chosen_sensor_by_seq)} sequences"
                )
            if ref_policy == "robust_mean":
                projection_info.append(
                    f"[INFO] Anchor-free target refs `{target_modality}` estimated by robust "
                    f"sequence means."
                )
            if ref_policy == "medoid":
                projection_info.append(
                    f"[INFO] Anchor-free target refs `{target_modality}` estimated by "
                    f"sequence medoid cameras."
                )
            if len(pred_target_seq_meta) > 0:
                seq0p = sorted(pred_target_seq_meta.keys())[0]
                p_idx, p_m = pred_target_seq_meta[seq0p]
                projection_info.append(
                    f"[INFO] Example seq `{seq0p}` Pred target ref: "
                    f"sample_idx={p_idx}, camera_idx={p_m}"
                )
            if len(gt_seq_meta) > 0:
                seq0 = sorted(gt_seq_meta.keys())[0]
                g_idx, g_m = gt_seq_meta[seq0]
                projection_info.append(
                    f"[INFO] Example seq `{seq0}` GT target ref: "
                    f"sample_idx={g_idx}, camera_idx={g_m}"
                )
            if reliability_source in {"temporal", "hybrid"}:
                for mod in active_fusion_modalities:
                    vals = list(temporal_reliability_by_modality.get(mod, {}).values())
                    if len(vals) == 0:
                        continue
                    projection_info.append(
                        f"[INFO] Temporal reliability `{mod}`: mean={np.mean(vals):.4f}, "
                        f"min={np.min(vals):.4f}, max={np.max(vals):.4f}"
                    )

            used_modality_counts = []
            rel_values_by_modality: Dict[str, List[float]] = {mod: [] for mod in active_fusion_modalities}
            dropped_no_camera_projection = 0
            worker_context = {
                "mode": "multi_sensor_rigfused",
                "pred_all": pred_all,
                "gt_all": gt_all,
                "sample_ids": sample_ids,
                "seq_group_names": seq_group_names,
                "num_samples": int(num_samples),
                "target_modality": target_modality,
                "active_fusion_modalities": active_fusion_modalities,
                "pred_extr_by_modality": pred_extr_by_modality,
                "pred_target_seq_extr": pred_target_seq_extr,
                "seq_rig_extr_by_modality": seq_rig_extr_by_modality,
                "gt_seq_extr_by_modality": gt_seq_extr_by_modality,
                "temporal_reliability_by_modality": temporal_reliability_by_modality,
                "reliability_source": reliability_source,
                "fusion_mode": fusion_mode,
                "cross_sensor_tau": float(args.cross_sensor_tau),
                "hard_gate_ratio": float(args.hard_gate_ratio),
            }
            for status, _i, seq_name, frame_idx, pred_proj, gt_proj, extra in _run_sample_worker_pool(
                context=worker_context,
                num_samples=num_samples,
                workers=int(args.workers),
                show_progress=show_progress,
                desc="transform keypoints",
            ):
                if status == "no_projection":
                    dropped_no_camera_projection += 1
                    continue
                if status != "ok":
                    continue

                used_mods, reliability_scores = extra
                pred_new.append(pred_proj)
                gt_new.append(gt_proj)
                used_modality_counts.append(int(used_mods))
                for mod, val in reliability_scores.items():
                    if mod in rel_values_by_modality:
                        rel_values_by_modality[mod].append(float(val))
                kept_seq_names.append(seq_name)
                kept_frame_indices.append(frame_idx)
                kept += 1

            if len(used_modality_counts) > 0:
                projection_info.append(
                    f"[INFO] Modalities used per sample: mean={np.mean(used_modality_counts):.2f}, "
                    f"min={np.min(used_modality_counts)}, max={np.max(used_modality_counts)}"
                )
            for mod in active_fusion_modalities:
                vals = rel_values_by_modality.get(mod, [])
                if len(vals) == 0:
                    continue
                projection_info.append(
                    f"[INFO] Reliability `{mod}`: mean={np.mean(vals):.4f}, "
                    f"min={np.min(vals):.4f}, max={np.max(vals):.4f}"
                )
            if dropped_no_camera_projection > 0:
                projection_info.append(
                    f"[INFO] Dropped samples without valid multi-sensor camera projection: "
                    f"{dropped_no_camera_projection}"
                )

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
    print(f"[INFO] Projection mode: {projection_mode}")
    for line in projection_info:
        print(line)
    print(
        f"[INFO] Smoothing: window={args.smooth_window}, target={args.smooth_on}, mode={args.smooth_mode}"
    )
    print(f"[INFO] Valid samples: {kept}/{num_samples}")
    print(f"[RESULT] MPJPE (fixed-sensor frame): {_format_mm(mpjpe_new)}")
    print(f"[RESULT] PA-MPJPE (fixed-sensor frame): {_format_mm(pampjpe_new)}")
    print(
        f"[RESULT] MPJPE (fixed-sensor frame, pelvis-centered @ joint {pelvis}): "
        f"{_format_mm(mpjpe_centered)}"
    )


if __name__ == "__main__":
    main()
