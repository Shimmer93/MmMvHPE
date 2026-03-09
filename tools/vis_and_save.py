"""
Visualize model inputs/outputs from a prediction pkl and save rendered images.

Implemented based on file comments:
- Input visualization: RGB image + point cloud.
- Output visualization: predicted keypoints, predicted SMPL mesh.
- Camera pose rendering is skipped for now.
- Supports deterministic index-range selection via --start/--end.
- Keeps fixed output layout/aspect and enforces equal 3D axes.
- For keypoints produced by eval_fixed_lidar_frame.py, can map sensor-space
  predictions back to world space via GT sensor camera extrinsics.

Example:
    uv run python tools/vis_and_save.py \
      --pred-file /root/MmMvHPE/logs/eccv26_humman/cross_camera_cam_json/HummanVIBEToken_test_predictions.pkl \
      --data-root /root/autodl-tmp/humman_cropped \
      --save-dir /root/autodl-tmp/logs/vis_examples \
      --start 0 --end 99 \
      --pred-keypoints-space sensor
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import re
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import torch
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from misc.pose_enc import pose_encoding_to_extri_intri
from misc.skeleton import COCOSkeleton, PanopticCOCO19Skeleton, SMPLSkeleton, SimpleCOCOSkeleton
from misc.utils import load
from models.smpl import SMPL
from eval_fixed_lidar_frame import (
    _build_sequence_reference_cameras_for_modality,
    _extract_camera_encoding,
    _get_camera_index,
    _get_sample_keypoints,
    _get_sample_lidar_center,
    _inverse_lidar_camera_center,
    _pose_encoding_to_extrinsic,
    _seq_name_from_sample_id,
    _transform_points,
)


_HUMMAN_ID_RE = re.compile(
    r"^(?P<seq>p\d+_a\d+)_rgb_(?P<rgb>kinect_\d{3}|iphone)_depth_"
    r"(?P<depth>kinect_\d{3}|iphone)_(?P<frame>\d+)$"
)


def _id_to_file_name_humman(sample_id: str, modality: str) -> str:
    match = _HUMMAN_ID_RE.match(sample_id)
    if match is None:
        raise ValueError(f"Unexpected HUMMAN sample_id format: {sample_id}")
    frame_idx = int(match.group("frame"))
    frame_token = f"{frame_idx + 1:06d}"
    if modality == "rgb":
        cam = match.group("rgb")
    elif modality == "depth":
        cam = match.group("depth")
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    return f"{match.group('seq')}_{cam}_{frame_token}"


def _depth_to_world_points(
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    max_points: int = 60000,
    min_depth: float = 1e-6,
) -> np.ndarray:
    if depth_m.ndim != 2:
        raise ValueError(f"Expected depth shape [H,W], got {depth_m.shape}")
    h, w = depth_m.shape
    z = depth_m.reshape(-1)
    valid = np.isfinite(z) & (z > min_depth)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    yy, xx = np.mgrid[0:h, 0:w]
    xx = xx.reshape(-1)[valid].astype(np.float32)
    yy = yy.reshape(-1)[valid].astype(np.float32)
    z = z[valid].astype(np.float32)

    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    x = (xx - cx) * z / max(fx, 1e-6)
    y = (yy - cy) * z / max(fy, 1e-6)
    cam_points = np.stack([x, y, z], axis=-1).astype(np.float32)

    # Camera <- World: X_cam = R * X_world + T
    # World <- Camera: X_world = (X_cam - T) * R
    t = translation.reshape(1, 3).astype(np.float32)
    world_points = (cam_points - t) @ rotation.astype(np.float32)

    if world_points.shape[0] > max_points:
        keep = np.random.choice(world_points.shape[0], size=max_points, replace=False)
        world_points = world_points[keep]
    return world_points.astype(np.float32)


def _load_humman_inputs(
    sample_id: str,
    data_root: str,
    max_pc_points: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    rgb_fn = _id_to_file_name_humman(sample_id, modality="rgb")
    depth_fn = _id_to_file_name_humman(sample_id, modality="depth")
    rgb_path = osp.join(data_root, "rgb", f"{rgb_fn}.jpg")
    depth_path = osp.join(data_root, "depth", f"{depth_fn}.png")

    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"Failed to read RGB image: {rgb_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth image: {depth_path}")
    depth = depth.astype(np.float32)
    # HUMMAN depth png is millimeter in this pipeline.
    if float(np.nanmax(depth)) > 100.0:
        depth = depth / 1000.0

    match = _HUMMAN_ID_RE.match(sample_id)
    if match is None:
        raise ValueError(f"Unexpected HUMMAN sample_id format: {sample_id}")
    seq_name = match.group("seq")
    depth_cam = match.group("depth")
    camera_file = osp.join(data_root, "cameras", f"{seq_name}_cameras.json")
    with open(camera_file, "r", encoding="utf-8") as f:
        cameras = json.load(f)

    if depth_cam.startswith("kinect"):
        cam_key = f"kinect_depth_{depth_cam.split('_')[1]}"
    else:
        cam_key = "iphone"
    if cam_key not in cameras:
        raise KeyError(f"Camera key `{cam_key}` not found in {camera_file}")
    cam = cameras[cam_key]
    k = np.asarray(cam["K"], dtype=np.float32)
    r = np.asarray(cam["R"], dtype=np.float32)
    t = np.asarray(cam["T"], dtype=np.float32).reshape(3)

    pc_world = _depth_to_world_points(
        depth_m=depth,
        intrinsic=k,
        rotation=r,
        translation=t,
        max_points=max_pc_points,
    )
    return rgb, pc_world, rgb_path


def _extract_sample_any(arr, sample_idx: int):
    if arr is None:
        return None
    if isinstance(arr, np.ndarray) and arr.dtype != object:
        if arr.ndim == 0:
            return arr
        if sample_idx >= arr.shape[0]:
            return None
        return arr[sample_idx]
    if sample_idx >= len(arr):
        return None
    return arr[sample_idx]


def _to_numpy_float(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    if x.size == 0:
        return None
    return x.astype(np.float32)


def _resolve_camera_key(data: dict, camera_key: str) -> str:
    if camera_key != "auto":
        if camera_key not in data:
            raise KeyError(f"Requested camera key `{camera_key}` not found in prediction file.")
        return camera_key
    for k in ("pred_cameras_stream", "pred_cameras", "gt_cameras_stream", "gt_cameras"):
        if k in data:
            return k
    raise KeyError("No camera encoding key found. Checked: pred/gt cameras (stream + non-stream).")


def _infer_bones(num_joints: int):
    if num_joints == SMPLSkeleton.num_joints:
        return SMPLSkeleton.bones
    if num_joints == PanopticCOCO19Skeleton.num_joints:
        return PanopticCOCO19Skeleton.bones
    if num_joints == COCOSkeleton.num_joints:
        return COCOSkeleton.bones
    if num_joints == SimpleCOCOSkeleton.num_joints:
        return SimpleCOCOSkeleton.bones
    return []


def _infer_skl_format(num_joints: int) -> str:
    if num_joints == SMPLSkeleton.num_joints:
        return "smpl"
    if num_joints == PanopticCOCO19Skeleton.num_joints:
        return "panoptic_coco19"
    if num_joints == COCOSkeleton.num_joints:
        return "coco"
    if num_joints == SimpleCOCOSkeleton.num_joints:
        return "simple_coco"
    return "unknown"


def _get_pelvis_from_keypoints(keypoints: Optional[np.ndarray], skl_format: str) -> Optional[np.ndarray]:
    if keypoints is None:
        return None
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] == 0:
        return None
    sf = str(skl_format).lower()
    if sf == "panoptic_coco19":
        if arr.shape[0] > 2:
            return arr[2]
    if sf == "coco":
        if arr.shape[0] > 12:
            return 0.5 * (arr[11] + arr[12])
    return arr[0]


def _get_pelvis_from_smpl_joints(joints: Optional[np.ndarray], skl_format: str) -> Optional[np.ndarray]:
    if joints is None:
        return None
    arr = np.asarray(joints, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] == 0:
        return None
    sf = str(skl_format).lower()
    if sf == "coco" and arr.shape[0] > 2:
        return 0.5 * (arr[1] + arr[2])
    return arr[0]


def _draw_skeleton_3d(
    ax,
    keypoints: np.ndarray,
    bone_color: str = "#444444",
    joint_color: str = "#222222",
    linewidth: float = 1.6,
    joint_size: float = 20.0,
):
    bones = _infer_bones(int(keypoints.shape[0]))
    for i, j in bones:
        if i < keypoints.shape[0] and j < keypoints.shape[0]:
            ax.plot(
                [keypoints[i, 0], keypoints[j, 0]],
                [keypoints[i, 1], keypoints[j, 1]],
                [keypoints[i, 2], keypoints[j, 2]],
                color=bone_color,
                linewidth=linewidth,
                alpha=0.95,
            )
    ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], s=joint_size, c=joint_color, depthshade=False)


def _draw_skeleton_2d(
    ax,
    keypoints_2d: np.ndarray,
    bone_color: str = "#444444",
    joint_color: str = "#222222",
    linewidth: float = 1.6,
    joint_size: float = 20.0,
):
    bones = _infer_bones(int(keypoints_2d.shape[0]))
    for i, j in bones:
        if i < keypoints_2d.shape[0] and j < keypoints_2d.shape[0]:
            ax.plot(
                [keypoints_2d[i, 0], keypoints_2d[j, 0]],
                [keypoints_2d[i, 1], keypoints_2d[j, 1]],
                color=bone_color,
                linewidth=linewidth,
                alpha=0.95,
            )
    ax.scatter(keypoints_2d[:, 0], keypoints_2d[:, 1], s=joint_size, c=joint_color, zorder=5)


def _set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    radius = 0.5 * max(x_range, y_range, z_range, 1e-6)
    ax.set_xlim3d([x_mid - radius, x_mid + radius])
    ax.set_ylim3d([y_mid - radius, y_mid + radius])
    ax.set_zlim3d([z_mid - radius, z_mid + radius])
    ax.set_box_aspect([1, 1, 1])


def _set_3d_limits_data_aspect(ax, points_3d_list: Sequence[np.ndarray], pad_ratio: float = 0.05):
    valid = []
    for p in points_3d_list:
        if p is None:
            continue
        arr = np.asarray(p, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 3:
            continue
        m = np.isfinite(arr).all(axis=1)
        if np.any(m):
            valid.append(arr[m, :3])
    if len(valid) == 0:
        return
    all_pts = np.concatenate(valid, axis=0)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    ax.set_xlim(mins[0] - pad_ratio * span[0], maxs[0] + pad_ratio * span[0])
    ax.set_ylim(mins[1] - pad_ratio * span[1], maxs[1] + pad_ratio * span[1])
    ax.set_zlim(mins[2] - pad_ratio * span[2], maxs[2] + pad_ratio * span[2])
    # Keep physical axis ratio from data ranges (not forced equal cube).
    ax.set_box_aspect(span.tolist())


def _pose_to_intrinsics(
    pose_encoding: np.ndarray,
    image_size_hw: Tuple[int, int],
    pose_encoding_type: str,
) -> np.ndarray:
    pe = torch.as_tensor(pose_encoding, dtype=torch.float32).view(1, 1, -1)
    _, intri = pose_encoding_to_extri_intri(
        pe,
        image_size_hw=image_size_hw,
        pose_encoding_type=pose_encoding_type,
        build_intrinsics=True,
    )
    return intri[0, 0].detach().cpu().numpy().astype(np.float32)


def _sensor_to_world(points_sensor: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    r = extrinsic[:, :3]
    t = extrinsic[:, 3]
    return points_sensor @ r.T + t[None, :]


def _world_to_sensor(points_world: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    r = extrinsic[:, :3]
    t = extrinsic[:, 3]
    return (points_world - t[None, :]) @ r


def _project_world_to_image(points_world: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    points_cam = _world_to_sensor(points_world, extrinsic)
    z = points_cam[:, 2]
    uvh = points_cam @ intrinsic.T
    u = uvh[:, 0] / np.maximum(z, 1e-6)
    v = uvh[:, 1] / np.maximum(z, 1e-6)
    out = np.stack([u, v], axis=-1).astype(np.float32)
    out[~np.isfinite(out).all(axis=1)] = np.nan
    return out


def _plot_mesh_3d(
    ax,
    vertices: Optional[np.ndarray],
    faces: np.ndarray,
    color,
    alpha: float,
):
    if vertices is None:
        return
    mesh = Poly3DCollection(vertices[faces], alpha=alpha, facecolor=color, edgecolor="none")
    ax.add_collection3d(mesh)


def _plot_mesh_projected_on_image(
    ax,
    vertices_world: Optional[np.ndarray],
    faces: np.ndarray,
    extrinsic_world_from_cam: Optional[np.ndarray],
    intrinsic: Optional[np.ndarray],
    color,
    alpha: float,
):
    if vertices_world is None or extrinsic_world_from_cam is None or intrinsic is None:
        return
    verts_cam = _world_to_sensor(vertices_world, extrinsic_world_from_cam)
    z = verts_cam[:, 2]
    uvh = verts_cam @ intrinsic.T
    uv = np.stack(
        [
            uvh[:, 0] / np.maximum(z, 1e-6),
            uvh[:, 1] / np.maximum(z, 1e-6),
        ],
        axis=-1,
    ).astype(np.float32)
    valid_faces = np.all((z[faces] > 1e-6) & np.isfinite(z[faces]), axis=1)
    if not np.any(valid_faces):
        return
    tri = uv[faces[valid_faces]]
    tri_depth = z[faces[valid_faces]].mean(axis=1)
    order = np.argsort(tri_depth)[::-1]
    rgba = mcolors.to_rgba(color, alpha=alpha)
    poly = PolyCollection(
        tri[order],
        facecolors=[rgba] * len(order),
        edgecolors="none",
        linewidths=0.0,
    )
    ax.add_collection(poly)


def _plot_mesh_2d_dims(
    ax,
    vertices: Optional[np.ndarray],
    faces: np.ndarray,
    dims: Tuple[int, int],
    color,
    alpha: float,
):
    if vertices is None:
        return
    dims = tuple(dims)
    tri = vertices[faces][:, :, list(dims)]
    depth_axis = list({0, 1, 2} - set(dims))[0]
    tri_depth = vertices[faces][:, :, depth_axis].mean(axis=1)
    order = np.argsort(tri_depth)[::-1]
    rgba = mcolors.to_rgba(color, alpha=alpha)
    poly = PolyCollection(
        tri[order],
        facecolors=[rgba] * len(order),
        edgecolors="none",
        linewidths=0.0,
    )
    ax.add_collection(poly)


def _set_2d_limits(ax, points_2d_list: Sequence[np.ndarray], pad_ratio: float = 0.08):
    valid = []
    for p in points_2d_list:
        if p is None:
            continue
        arr = np.asarray(p, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        m = np.isfinite(arr).all(axis=1)
        if np.any(m):
            valid.append(arr[m, :2])
    if len(valid) == 0:
        return
    all_pts = np.concatenate(valid, axis=0)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    ax.set_xlim(mins[0] - pad_ratio * span[0], maxs[0] + pad_ratio * span[0])
    ax.set_ylim(mins[1] - pad_ratio * span[1], maxs[1] + pad_ratio * span[1])
    ax.set_aspect("equal")


def _clean_axis_2d(ax):
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_frame_on(False)


def _clean_axis_3d(ax):
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")


def _load_smpl_vertices(
    smpl_model: SMPL,
    smpl_params: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if smpl_params is None:
        return None, None
    smpl_params = np.asarray(smpl_params, dtype=np.float32).reshape(-1)
    if smpl_params.shape[0] < 82:
        return None, None

    # Support both plain SMPL (82 = 72 pose + 10 betas) and
    # camera-prefixed layouts (85 = 3 camera + 82 SMPL).
    if smpl_params.shape[0] >= 85:
        smpl_params = smpl_params[-82:]
    else:
        smpl_params = smpl_params[:82]

    pose_body = smpl_params[3:72]
    global_orient = smpl_params[:3]
    betas = smpl_params[72:82]
    translation = np.zeros((3,), dtype=np.float32)

    with torch.no_grad():
        pose_tensor = torch.from_numpy(pose_body).float().unsqueeze(0)
        go_tensor = torch.from_numpy(global_orient).float().unsqueeze(0)
        full_pose = torch.cat([go_tensor, pose_tensor], dim=1)
        beta_tensor = torch.from_numpy(betas).float().unsqueeze(0)
        trans_tensor = torch.from_numpy(translation).float().unsqueeze(0)
        verts, joints = smpl_model(
            th_pose_axisang=full_pose,
            th_betas=beta_tensor,
            th_trans=trans_tensor,
        )

    verts = verts[0].detach().cpu().numpy().astype(np.float32)
    joints = joints[0].detach().cpu().numpy().astype(np.float32)
    return verts, joints


def _get_camera_extrinsic_and_intrinsic(
    data: dict,
    camera_arr,
    camera_key: str,
    sample_idx: int,
    modality: str,
    sensor_idx: int,
    pose_encoding_type: str,
    image_size_hw: Tuple[int, int],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    use_stream_index = str(camera_key).endswith("_stream")
    cam_idx = _get_camera_index(
        data=data,
        sample_idx=sample_idx,
        modality=modality,
        fallback_idx=0,
        use_stream_index=use_stream_index,
        sensor_idx=sensor_idx,
    )
    cam_enc = _extract_camera_encoding(camera_arr, sample_idx, cam_idx)
    if modality == "lidar":
        center = _get_sample_lidar_center(data, sample_idx)
        cam_enc = _inverse_lidar_camera_center(cam_enc, center)
    if cam_enc is None:
        return None, None, None

    cam_enc = np.asarray(cam_enc, dtype=np.float32).reshape(-1)
    if cam_enc.shape[0] < 7 or not np.isfinite(cam_enc).all():
        return None, None, None

    try:
        extr = _pose_encoding_to_extrinsic(cam_enc, pose_encoding_type).astype(np.float32)
    except Exception:
        return None, None, None

    intr = None
    if modality in {"rgb", "depth"}:
        try:
            intr = _pose_to_intrinsics(cam_enc, image_size_hw=image_size_hw, pose_encoding_type=pose_encoding_type)
        except Exception:
            intr = None
    return extr, intr, cam_enc


def _to_save_name(sample_id: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(sample_id))


def visualize_and_save(
    pred_file: str,
    data_root: str,
    save_dir: str,
    start: int,
    end: int,
    dataset: str,
    pose_encoding_type: str,
    camera_key: str,
    pred_keypoints_key: str,
    gt_keypoints_key: str,
    smpl_params_key: str,
    smpl_model_path: str,
    pred_keypoints_space: str,
    lidar_frame_mode: str,
    sensor_modality: str,
    sensor_idx: int,
    max_pc_points: int,
    skl_format: str,
):
    os.makedirs(save_dir, exist_ok=True)
    data = load(pred_file)
    sample_ids = data.get("sample_ids", None)
    if sample_ids is None:
        raise KeyError("Prediction file must contain `sample_ids`.")
    total = len(sample_ids)
    if total == 0:
        raise ValueError("Empty `sample_ids` in prediction file.")

    if dataset != "humman_preproc":
        raise NotImplementedError(f"Unsupported dataset `{dataset}`. Currently only `humman_preproc` is supported.")

    cam_key = _resolve_camera_key(data, camera_key)
    camera_arr = data.get(cam_key, None)
    gt_cam_key = None
    for k in ("gt_cameras_stream", "gt_cameras"):
        if k in data:
            gt_cam_key = k
            break
    gt_camera_arr = data.get(gt_cam_key, None) if gt_cam_key is not None else None

    pred_all = data.get(pred_keypoints_key, None)
    gt_all = data.get(gt_keypoints_key, None)
    smpl_params_all = data.get(smpl_params_key, None)
    gt_smpl_params_all = data.get("gt_smpl_params", None)

    if pred_all is None and data.get("pred_smpl_keypoints", None) is not None:
        pred_all = data["pred_smpl_keypoints"]
    if pred_all is None:
        raise KeyError(
            f"Missing `{pred_keypoints_key}` and fallback `pred_smpl_keypoints` in prediction file."
        )

    gt_seq_extr_by_seq = {}
    use_seq_fixed = str(lidar_frame_mode).lower() == "seq_fixed"
    if use_seq_fixed:
        try:
            if gt_camera_arr is not None:
                gt_seq_cam_enc, _ = _build_sequence_reference_cameras_for_modality(
                    data=data,
                    camera_key=gt_cam_key,
                    sample_ids=sample_ids,
                    num_samples=total,
                    modality=sensor_modality,
                    fallback_modality_idx=0,
                    sensor_idx=sensor_idx,
                    show_progress=False,
                )
                gt_seq_extr_by_seq = {
                    seq: _pose_encoding_to_extrinsic(cam_enc, pose_encoding_type).astype(np.float32)
                    for seq, cam_enc in gt_seq_cam_enc.items()
                }
            print(
                f"[INFO] seq-fixed {sensor_modality} refs (GT): {len(gt_seq_extr_by_seq)}"
            )
        except Exception as exc:
            use_seq_fixed = False
            print(
                f"[WARN] failed to build seq-fixed {sensor_modality} refs ({exc}); "
                "fallback to per-sample transform."
            )
    start = int(start)
    end = int(end)
    if start < 0:
        raise ValueError(f"`--start` must be >= 0, got {start}.")
    if end == -1:
        end = total - 1
    elif end < -1:
        raise ValueError(f"`--end` must be >= 0 or -1 (use -1 for last sample), got {end}.")
    if start >= total:
        raise ValueError(f"`--start`={start} out of range for total samples={total}.")
    if end >= total:
        raise ValueError(f"`--end`={end} out of range for total samples={total}.")
    if start > end:
        raise ValueError(f"`--start` must be <= `--end`, got start={start}, end={end}.")

    # Inclusive range [start, end].
    chosen = list(range(start, end + 1))

    smpl_model = SMPL(model_path=smpl_model_path)
    faces = smpl_model.th_faces.detach().cpu().numpy()
    # Paper-friendly, colorblind-safe style: warm=pred, cool=gt.
    pred_bone_color = "#D55E00"
    pred_joint_color = "#A73E00"
    gt_bone_color = "#0072B2"
    gt_joint_color = "#004B78"
    pred_mesh_color = "#F4C095"
    gt_mesh_color = "#9CC4E4"

    for idx in tqdm(chosen, desc="visualize", total=len(chosen)):
        sample_id = sample_ids[idx]
        try:
            rgb, pc_world, rgb_src_path = _load_humman_inputs(
                sample_id=sample_id,
                data_root=data_root,
                max_pc_points=max_pc_points,
            )
        except Exception as exc:
            print(f"[WARN] skip sample {sample_id}: failed to load input data ({exc})")
            continue

        h, w = rgb.shape[:2]
        pred_kps = _get_sample_keypoints(pred_all, idx)
        pred_kps = _to_numpy_float(pred_kps)
        if pred_kps is None or pred_kps.ndim != 2 or pred_kps.shape[-1] != 3:
            print(f"[WARN] skip sample {sample_id}: invalid predicted keypoints shape.")
            continue
        sample_skl_format = str(skl_format).lower()
        if sample_skl_format == "auto":
            sample_skl_format = _infer_skl_format(int(pred_kps.shape[0]))

        gt_kps = _get_sample_keypoints(gt_all, idx)
        gt_kps = _to_numpy_float(gt_kps)
        if gt_kps is not None and (gt_kps.ndim != 2 or gt_kps.shape[-1] != 3):
            gt_kps = None

        seq_name = _seq_name_from_sample_id(sample_id)
        seq_gt_extr = gt_seq_extr_by_seq.get(seq_name, None) if use_seq_fixed else None
        gt_sensor_extr = None
        if gt_camera_arr is not None:
            gt_sensor_extr, _, _ = _get_camera_extrinsic_and_intrinsic(
                data=data,
                camera_arr=gt_camera_arr,
                camera_key=gt_cam_key,
                sample_idx=idx,
                modality=sensor_modality,
                sensor_idx=sensor_idx,
                pose_encoding_type=pose_encoding_type,
                image_size_hw=(h, w),
            )
        gt_world_extr = seq_gt_extr if seq_gt_extr is not None else gt_sensor_extr

        pred_kps_world = pred_kps.copy()
        pred_world_extr = None
        sensor_extr, _, _ = _get_camera_extrinsic_and_intrinsic(
            data=data,
            camera_arr=camera_arr,
            camera_key=cam_key,
            sample_idx=idx,
            modality=sensor_modality,
            sensor_idx=sensor_idx,
            pose_encoding_type=pose_encoding_type,
            image_size_hw=(h, w),
        )
        if pred_keypoints_space == "sensor":
            if gt_world_extr is not None:
                pred_kps_world = _transform_points(pred_kps, gt_world_extr)
                pred_world_extr = gt_world_extr
            else:
                print(
                    f"[WARN] skip sample {sample_id}: GT {sensor_modality} camera unavailable; "
                    "cannot map sensor-space predictions to world."
                )
                continue
        elif pred_keypoints_space == "auto":
            if pred_keypoints_key == "pred_keypoints":
                if gt_world_extr is not None:
                    pred_kps_world = _transform_points(pred_kps, gt_world_extr)
                    pred_world_extr = gt_world_extr
                else:
                    print(
                        f"[WARN] skip sample {sample_id}: GT {sensor_modality} camera unavailable; "
                        "cannot map sensor-space predictions to world."
                    )
                    continue

        gt_kps_world = gt_kps
        if gt_kps is not None and gt_world_extr is not None:
            gt_kps_world = _transform_points(gt_kps, gt_world_extr)

        pred_smpl_kps_world = None
        pred_smpl_kps_all = data.get("pred_smpl_keypoints", None)
        pred_smpl_kps = _get_sample_keypoints(pred_smpl_kps_all, idx)
        pred_smpl_kps = _to_numpy_float(pred_smpl_kps)
        if pred_smpl_kps is not None and pred_smpl_kps.ndim == 2 and pred_smpl_kps.shape[-1] == 3:
            pred_smpl_kps_world = pred_smpl_kps.copy()
            if pred_keypoints_space == "sensor":
                if gt_world_extr is not None:
                    pred_smpl_kps_world = _transform_points(pred_smpl_kps, gt_world_extr)
            elif pred_keypoints_space == "auto":
                if gt_world_extr is not None and pred_keypoints_key == "pred_keypoints":
                    pred_smpl_kps_world = _transform_points(pred_smpl_kps, gt_world_extr)

        rgb_extr, rgb_intr, _ = _get_camera_extrinsic_and_intrinsic(
            data=data,
            camera_arr=camera_arr,
            camera_key=cam_key,
            sample_idx=idx,
            modality="rgb",
            sensor_idx=0,
            pose_encoding_type=pose_encoding_type,
            image_size_hw=(h, w),
        )

        if rgb_extr is not None and rgb_intr is not None:
            pred_uv = _project_world_to_image(pred_kps_world, rgb_extr, rgb_intr)
            gt_uv = _project_world_to_image(gt_kps_world, rgb_extr, rgb_intr) if gt_kps_world is not None else None
        else:
            pred_uv = np.full((pred_kps_world.shape[0], 2), np.nan, dtype=np.float32)
            gt_uv = None

        vis_sensor_extr = gt_world_extr if gt_world_extr is not None else sensor_extr
        if vis_sensor_extr is not None:
            pred_kps_sensor = _world_to_sensor(pred_kps_world, vis_sensor_extr)
            gt_kps_sensor = _world_to_sensor(gt_kps_world, vis_sensor_extr) if gt_kps_world is not None else None
            pc_sensor = _world_to_sensor(pc_world, vis_sensor_extr) if pc_world.size > 0 else pc_world
        else:
            pred_kps_sensor = pred_kps_world
            gt_kps_sensor = gt_kps_world
            pc_sensor = pc_world

        smpl_params = _extract_sample_any(smpl_params_all, idx)
        smpl_params = _to_numpy_float(smpl_params)
        mesh_vertices, mesh_joints = _load_smpl_vertices(smpl_model, smpl_params)
        # Keep mesh/joints in the same frame as pred_kps_world.
        if pred_world_extr is not None and mesh_vertices is not None:
            mesh_vertices = _transform_points(mesh_vertices, pred_world_extr)
        if pred_world_extr is not None and mesh_joints is not None:
            mesh_joints = _transform_points(mesh_joints, pred_world_extr)

        # Use same pelvis-alignment strategy as misc/vis.py.
        pred_pelvis_kp = _get_pelvis_from_keypoints(pred_kps_world, sample_skl_format)
        pred_pelvis_mesh = _get_pelvis_from_smpl_joints(mesh_joints, sample_skl_format)
        if pred_smpl_kps_world is not None:
            pelvis_from_smpl_kp = _get_pelvis_from_keypoints(pred_smpl_kps_world, sample_skl_format)
            if pelvis_from_smpl_kp is not None:
                pred_pelvis_mesh = pelvis_from_smpl_kp
        if mesh_vertices is not None and pred_pelvis_kp is not None and pred_pelvis_mesh is not None:
            mesh_vertices = mesh_vertices + (pred_pelvis_kp.reshape(1, 3) - pred_pelvis_mesh.reshape(1, 3))
            if mesh_joints is not None:
                mesh_joints = mesh_joints + (pred_pelvis_kp.reshape(1, 3) - pred_pelvis_mesh.reshape(1, 3))

        gt_smpl_params = _extract_sample_any(gt_smpl_params_all, idx)
        gt_smpl_params = _to_numpy_float(gt_smpl_params)
        gt_mesh_vertices, gt_mesh_joints = _load_smpl_vertices(smpl_model, gt_smpl_params)
        if gt_world_extr is not None and gt_mesh_vertices is not None:
            gt_mesh_vertices = _transform_points(gt_mesh_vertices, gt_world_extr)
        if gt_world_extr is not None and gt_mesh_joints is not None:
            gt_mesh_joints = _transform_points(gt_mesh_joints, gt_world_extr)
        gt_pelvis_kp = _get_pelvis_from_keypoints(gt_kps_world, sample_skl_format)
        gt_pelvis_mesh = _get_pelvis_from_smpl_joints(gt_mesh_joints, sample_skl_format)
        if gt_mesh_vertices is not None and gt_pelvis_kp is not None and gt_pelvis_mesh is not None:
            gt_mesh_vertices = gt_mesh_vertices + (gt_pelvis_kp.reshape(1, 3) - gt_pelvis_mesh.reshape(1, 3))
            if gt_mesh_joints is not None:
                gt_mesh_joints = gt_mesh_joints + (gt_pelvis_kp.reshape(1, 3) - gt_pelvis_mesh.reshape(1, 3))

        mesh_vertices_sensor = None
        if mesh_vertices is not None:
            if vis_sensor_extr is not None:
                mesh_vertices_sensor = _world_to_sensor(mesh_vertices, vis_sensor_extr)
            else:
                mesh_vertices_sensor = mesh_vertices.copy()
        gt_mesh_vertices_sensor = None
        if gt_mesh_vertices is not None:
            if vis_sensor_extr is not None:
                gt_mesh_vertices_sensor = _world_to_sensor(gt_mesh_vertices, vis_sensor_extr)
            else:
                gt_mesh_vertices_sensor = gt_mesh_vertices.copy()

        sample_dir = osp.join(save_dir, _to_save_name(sample_id))
        os.makedirs(sample_dir, exist_ok=True)

        # Save raw RGB frame only (no overlays).
        try:
            shutil.copy2(rgb_src_path, osp.join(sample_dir, "rgb.jpg"))
        except Exception:
            # Fallback: write from loaded array if direct copy is unavailable.
            cv2.imwrite(osp.join(sample_dir, "rgb.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # Sensor 3D: point cloud only.
        def _save_sensor_pointcloud():
            fig = plt.figure(figsize=(6.2, 6.0), dpi=130)
            ax = fig.add_subplot(111, projection="3d")
            if pc_sensor.size > 0:
                ax.scatter(pc_sensor[:, 0], pc_sensor[:, 1], pc_sensor[:, 2], s=0.3, c="gray", alpha=0.35)
                _set_3d_limits_data_aspect(ax, [pc_sensor], pad_ratio=0.03)
            _clean_axis_3d(ax)
            fig.tight_layout()
            fig.savefig(osp.join(sample_dir, "sensor_3d_pointcloud.png"), dpi=130)
            plt.close(fig)

        def _save_world_3d(mode: str):
            fig = plt.figure(figsize=(6.2, 6.0), dpi=130)
            ax = fig.add_subplot(111, projection="3d")
            if mode in {"pred", "overlay"}:
                _plot_mesh_3d(ax, mesh_vertices, faces, pred_mesh_color, alpha=0.32)
                _draw_skeleton_3d(
                    ax,
                    pred_kps_world,
                    bone_color=pred_bone_color,
                    joint_color=pred_joint_color,
                    linewidth=1.7,
                    joint_size=28.0,
                )
            if mode in {"gt", "overlay"} and gt_kps_world is not None:
                _plot_mesh_3d(ax, gt_mesh_vertices, faces, gt_mesh_color, alpha=0.28)
                _draw_skeleton_3d(
                    ax,
                    gt_kps_world,
                    bone_color=gt_bone_color,
                    joint_color=gt_joint_color,
                    linewidth=1.6,
                    joint_size=26.0,
                )
            _set_axes_equal(ax)
            _clean_axis_3d(ax)
            fig.tight_layout()
            fig.savefig(osp.join(sample_dir, f"world_3d_{mode}.png"), dpi=130)
            plt.close(fig)

        def _save_world_2d_view(mode: str, dims: Tuple[int, int], name: str):
            fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=130)
            points_for_limit = []
            if mode in {"pred", "overlay"}:
                _plot_mesh_2d_dims(ax, mesh_vertices, faces, dims=dims, color=pred_mesh_color, alpha=0.30)
                pred_2d = pred_kps_world[:, list(dims)]
                _draw_skeleton_2d(
                    ax,
                    pred_2d,
                    bone_color=pred_bone_color,
                    joint_color=pred_joint_color,
                    linewidth=1.6,
                    joint_size=30.0,
                )
                if mesh_vertices is not None:
                    points_for_limit.append(mesh_vertices[:, list(dims)])
                points_for_limit.append(pred_2d)
            if mode in {"gt", "overlay"} and gt_kps_world is not None:
                _plot_mesh_2d_dims(ax, gt_mesh_vertices, faces, dims=dims, color=gt_mesh_color, alpha=0.28)
                gt_2d = gt_kps_world[:, list(dims)]
                _draw_skeleton_2d(
                    ax,
                    gt_2d,
                    bone_color=gt_bone_color,
                    joint_color=gt_joint_color,
                    linewidth=1.5,
                    joint_size=28.0,
                )
                if gt_mesh_vertices is not None:
                    points_for_limit.append(gt_mesh_vertices[:, list(dims)])
                points_for_limit.append(gt_2d)
            _set_2d_limits(ax, points_for_limit)
            if name == "XY":
                ax.invert_yaxis()
            _clean_axis_2d(ax)
            fig.tight_layout()
            fig.savefig(osp.join(sample_dir, f"world_{name.lower()}_{mode}.png"), dpi=130)
            plt.close(fig)

        _save_sensor_pointcloud()
        for mode_name in ("pred", "gt", "overlay"):
            _save_world_3d(mode_name)
            _save_world_2d_view(mode_name, (0, 1), "XY")
            _save_world_2d_view(mode_name, (0, 2), "XZ")
            _save_world_2d_view(mode_name, (1, 2), "YZ")

    print(f"Saved {len(chosen)} visualization images to: {save_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize predictions and save rendered images.")
    parser.add_argument("--pred-file", type=str, required=True, help="Path to prediction pkl file.")
    parser.add_argument("--data-root", type=str, default="/root/autodl-tmp/humman_cropped")
    parser.add_argument("--save-dir", type=str, default="visualization_results")
    parser.add_argument("--dataset", type=str, default="humman_preproc", choices=["humman_preproc"])
    parser.add_argument("--start", type=int, default=0, help="Start sample index (inclusive).")
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="End sample index (inclusive). Use -1 for the last sample.",
    )
    parser.add_argument("--pose-encoding-type", type=str, default="absT_quaR_FoV")
    parser.add_argument("--camera-key", type=str, default="auto")
    parser.add_argument("--pred-keypoints-key", type=str, default="pred_keypoints")
    parser.add_argument("--gt-keypoints-key", type=str, default="gt_keypoints")
    parser.add_argument("--smpl-params-key", type=str, default="pred_smpl_params")
    parser.add_argument("--smpl-model-path", type=str, default="weights/smpl/SMPL_NEUTRAL.pkl")
    parser.add_argument(
        "--skl-format",
        type=str,
        default="auto",
        choices=["auto", "smpl", "coco", "panoptic_coco19", "simple_coco"],
        help="Skeleton format for pelvis anchoring. `auto` infers from joint count.",
    )
    parser.add_argument(
        "--pred-keypoints-space",
        type=str,
        default="sensor",
        choices=["sensor", "world", "auto"],
        help="Coordinate space of predicted keypoints in pred-keypoints-key.",
    )
    parser.add_argument(
        "--lidar-frame-mode",
        type=str,
        default="seq_fixed",
        choices=["seq_fixed", "per_sample"],
        help="`seq_fixed` matches eval_fixed_lidar_frame.py sequence-fixed LiDAR mapping.",
    )
    parser.add_argument("--sensor-modality", type=str, default="lidar", choices=["lidar", "depth", "rgb"])
    parser.add_argument("--sensor-idx", type=int, default=0)
    parser.add_argument("--max-pc-points", type=int, default=60000)
    return parser


def main():
    args = build_parser().parse_args()
    visualize_and_save(
        pred_file=args.pred_file,
        data_root=args.data_root,
        save_dir=args.save_dir,
        start=args.start,
        end=args.end,
        dataset=args.dataset,
        pose_encoding_type=args.pose_encoding_type,
        camera_key=args.camera_key,
        pred_keypoints_key=args.pred_keypoints_key,
        gt_keypoints_key=args.gt_keypoints_key,
        smpl_params_key=args.smpl_params_key,
        smpl_model_path=args.smpl_model_path,
        pred_keypoints_space=args.pred_keypoints_space,
        lidar_frame_mode=args.lidar_frame_mode,
        sensor_modality=args.sensor_modality,
        sensor_idx=args.sensor_idx,
        max_pc_points=args.max_pc_points,
        skl_format=args.skl_format,
    )


if __name__ == "__main__":
    main()
