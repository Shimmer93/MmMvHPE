"""
Visualize GT HUMMAN sequence data across all available camera views.

For each selected frame, saves:
1) RGB image per RGB view
2) Depth-derived point cloud per depth view
3) GT 3D skeleton in sensor coordinates per depth view
4) GT 2D skeleton projection per RGB view (skeleton-only canvas)
5) Canonical GT 3D skeleton (pelvis-centered, root-rotation removed)

Example:
    uv run python tools/vis_gt_sequence.py \
      --sequence p000438_a000040 \
      --data-root /root/autodl-tmp/humman_cropped \
      --save-dir /root/autodl-tmp/logs/vis_gt_sequence \
      --start-frame 0 --end-frame 20 --frame-step 5
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import re
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from misc.skeleton import COCOSkeleton, PanopticCOCO19Skeleton, SMPLSkeleton, SimpleCOCOSkeleton

_SEQ_RE = re.compile(r"^p\d+_a\d+$")


def _axis_angle_to_matrix_np(axis_angle: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = axis_angle / angle
    x, y, z = axis
    k = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float32,
    )
    eye = np.eye(3, dtype=np.float32)
    return eye + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


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


def _camera_name_to_key(camera_name: str, modality: str) -> str:
    if camera_name.startswith("kinect"):
        suffix = camera_name.split("_")[1]
        if modality == "rgb":
            return f"kinect_color_{suffix}"
        if modality == "depth":
            return f"kinect_depth_{suffix}"
    return "iphone"


def _extract_camera_names(cameras: Dict[str, dict], modality: str) -> List[str]:
    names = []
    if modality == "rgb":
        prefix = "kinect_color_"
    elif modality == "depth":
        prefix = "kinect_depth_"
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    for key in cameras:
        if key.startswith(prefix):
            suffix = key.split("_")[-1]
            names.append(f"kinect_{suffix}")
        elif key == "iphone":
            names.append("iphone")
    names = sorted(set(names))
    if "iphone" in names:
        names = [x for x in names if x != "iphone"] + ["iphone"]
    return names


def _to_frame_token(frame_idx: int) -> str:
    return f"{int(frame_idx) + 1:06d}"


def _rgb_path(data_root: str, seq: str, cam: str, frame_idx: int) -> str:
    return osp.join(data_root, "rgb", f"{seq}_{cam}_{_to_frame_token(frame_idx)}.jpg")


def _depth_path(data_root: str, seq: str, cam: str, frame_idx: int) -> str:
    return osp.join(data_root, "depth", f"{seq}_{cam}_{_to_frame_token(frame_idx)}.png")


def _load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read RGB image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_depth_m(path: str) -> np.ndarray:
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth image: {path}")
    depth = depth.astype(np.float32)
    if float(np.nanmax(depth)) > 100.0:
        depth = depth / 1000.0
    return depth


def _depth_to_sensor_points(
    depth_m: np.ndarray,
    intrinsic: np.ndarray,
    max_points: int,
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
    points = np.stack([x, y, z], axis=-1).astype(np.float32)

    if points.shape[0] > max_points:
        keep = np.random.choice(points.shape[0], size=max_points, replace=False)
        points = points[keep]
    return points


def _world_to_sensor(points_world: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return points_world @ rotation.T + translation.reshape(1, 3)


def _project_world_to_image(
    points_world: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    points_cam = _world_to_sensor(points_world, rotation, translation)
    z = points_cam[:, 2]
    uvh = points_cam @ intrinsic.T
    u = uvh[:, 0] / np.maximum(z, 1e-6)
    v = uvh[:, 1] / np.maximum(z, 1e-6)
    uv = np.stack([u, v], axis=-1).astype(np.float32)
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(z) & (z > 1e-6)
    uv[~valid] = np.nan
    return uv


def _draw_skeleton_3d(
    ax,
    keypoints: np.ndarray,
    bone_color: str = "#0072B2",
    joint_color: str = "#004B78",
    linewidth: float = 1.6,
    joint_size: float = 18.0,
):
    bones = _infer_bones(int(keypoints.shape[0]))
    for i, j in bones:
        if i < keypoints.shape[0] and j < keypoints.shape[0]:
            p0 = keypoints[i]
            p1 = keypoints[j]
            if np.isfinite(p0).all() and np.isfinite(p1).all():
                ax.plot(
                    [p0[0], p1[0]],
                    [p0[1], p1[1]],
                    [p0[2], p1[2]],
                    color=bone_color,
                    linewidth=linewidth,
                    alpha=0.95,
                )
    m = np.isfinite(keypoints).all(axis=1)
    if np.any(m):
        ax.scatter(
            keypoints[m, 0],
            keypoints[m, 1],
            keypoints[m, 2],
            s=joint_size,
            c=joint_color,
            depthshade=False,
        )


def _draw_skeleton_2d(
    ax,
    keypoints_2d: np.ndarray,
    bone_color: str = "#0072B2",
    joint_color: str = "#004B78",
    linewidth: float = 1.8,
    joint_size: float = 15.0,
):
    bones = _infer_bones(int(keypoints_2d.shape[0]))
    for i, j in bones:
        if i < keypoints_2d.shape[0] and j < keypoints_2d.shape[0]:
            p0 = keypoints_2d[i]
            p1 = keypoints_2d[j]
            if np.isfinite(p0).all() and np.isfinite(p1).all():
                ax.plot(
                    [p0[0], p1[0]],
                    [p0[1], p1[1]],
                    color=bone_color,
                    linewidth=linewidth,
                    alpha=0.95,
                )
    m = np.isfinite(keypoints_2d).all(axis=1)
    if np.any(m):
        ax.scatter(keypoints_2d[m, 0], keypoints_2d[m, 1], s=joint_size, c=joint_color, zorder=5)


def _set_3d_limits_data_aspect(ax, points_3d_list: Sequence[np.ndarray], pad_ratio: float = 0.06):
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
    ax.set_box_aspect(span.tolist())


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


def _camera_params(cameras: Dict[str, dict], camera_name: str, modality: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = _camera_name_to_key(camera_name, modality)
    if key not in cameras:
        raise KeyError(f"Camera key `{key}` missing for camera `{camera_name}` modality `{modality}`")
    cam = cameras[key]
    k = np.asarray(cam["K"], dtype=np.float32)
    r = np.asarray(cam["R"], dtype=np.float32)
    t = np.asarray(cam["T"], dtype=np.float32).reshape(3)
    if k.shape != (3, 3) or r.shape != (3, 3):
        raise ValueError(f"Invalid camera matrix shape for `{key}`: K={k.shape}, R={r.shape}")
    return k, r, t


def _canonicalize_world_keypoints(
    keypoints_world: np.ndarray,
    global_orient: np.ndarray,
) -> np.ndarray:
    if keypoints_world.ndim != 2 or keypoints_world.shape[1] != 3:
        raise ValueError(f"Expected keypoints shape [J,3], got {keypoints_world.shape}")
    pelvis = np.asarray(keypoints_world[0], dtype=np.float32).reshape(3)
    rot = _axis_angle_to_matrix_np(np.asarray(global_orient, dtype=np.float32).reshape(3))
    return (rot.T @ (keypoints_world.astype(np.float32) - pelvis).T).T.astype(np.float32)


def _save_rgb_view(
    out_dir: str,
    seq: str,
    frame_idx: int,
    data_root: str,
    camera_name: str,
):
    src = _rgb_path(data_root, seq, camera_name, frame_idx)
    if not osp.exists(src):
        return
    dst = osp.join(out_dir, f"{camera_name}.jpg")
    shutil.copy2(src, dst)


def _save_depth_pc_view(
    out_dir: str,
    seq: str,
    frame_idx: int,
    data_root: str,
    camera_name: str,
    cameras: Dict[str, dict],
    max_pc_points: int,
):
    path = _depth_path(data_root, seq, camera_name, frame_idx)
    if not osp.exists(path):
        return
    depth = _load_depth_m(path)
    k, _, _ = _camera_params(cameras, camera_name, "depth")
    pc_sensor = _depth_to_sensor_points(depth, k, max_points=max_pc_points)

    fig = plt.figure(figsize=(4.6, 4.2), dpi=135)
    ax = fig.add_subplot(111, projection="3d")
    if pc_sensor.size > 0:
        ax.scatter(
            pc_sensor[:, 0],
            pc_sensor[:, 1],
            pc_sensor[:, 2],
            s=0.35,
            c="gray",
            alpha=0.28,
        )
        _set_3d_limits_data_aspect(ax, [pc_sensor], pad_ratio=0.03)
    _clean_axis_3d(ax)
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, f"{camera_name}.png"), dpi=135)
    plt.close(fig)


def _save_skeleton_3d_view(
    out_dir: str,
    keypoints_world: np.ndarray,
    camera_name: str,
    cameras: Dict[str, dict],
):
    _, r, t = _camera_params(cameras, camera_name, "depth")
    kps_sensor = _world_to_sensor(keypoints_world, r, t)

    fig = plt.figure(figsize=(4.6, 4.2), dpi=135)
    ax = fig.add_subplot(111, projection="3d")
    _draw_skeleton_3d(
        ax=ax,
        keypoints=kps_sensor,
        bone_color="#0072B2",
        joint_color="#004B78",
        linewidth=1.7,
        joint_size=18.0,
    )
    _set_3d_limits_data_aspect(ax, [kps_sensor], pad_ratio=0.10)
    _clean_axis_3d(ax)
    fig.tight_layout()
    fig.savefig(osp.join(out_dir, f"{camera_name}.png"), dpi=135)
    plt.close(fig)


def _save_skeleton_2d_view(
    out_dir: str,
    seq: str,
    frame_idx: int,
    data_root: str,
    keypoints_world: np.ndarray,
    camera_name: str,
    cameras: Dict[str, dict],
):
    path = _rgb_path(data_root, seq, camera_name, frame_idx)
    if not osp.exists(path):
        return
    rgb = _load_rgb(path)
    h, w = rgb.shape[:2]
    k, r, t = _camera_params(cameras, camera_name, "rgb")
    uv = _project_world_to_image(keypoints_world, r, t, k)

    fig, ax = plt.subplots(figsize=(w / 180.0, h / 180.0), dpi=180)
    ax.set_facecolor("white")
    _draw_skeleton_2d(
        ax=ax,
        keypoints_2d=uv,
        bone_color="#D55E00",
        joint_color="#A73E00",
        linewidth=1.9,
        joint_size=16.0,
    )
    ax.set_xlim(0.0, float(w - 1))
    ax.set_ylim(float(h - 1), 0.0)
    _clean_axis_2d(ax)
    fig.tight_layout(pad=0.0)
    fig.savefig(osp.join(out_dir, f"{camera_name}.png"), dpi=180)
    plt.close(fig)


def _save_canonical_skeleton_3d(
    out_path: str,
    keypoints_canonical: np.ndarray,
):
    fig = plt.figure(figsize=(4.8, 4.4), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    _draw_skeleton_3d(
        ax=ax,
        keypoints=keypoints_canonical,
        bone_color="#4E79A7",
        joint_color="#2F4B6E",
        linewidth=1.8,
        joint_size=20.0,
    )
    _set_3d_limits_data_aspect(ax, [keypoints_canonical], pad_ratio=0.15)
    _clean_axis_3d(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def visualize_gt_sequence(
    data_root: str,
    sequence: str,
    save_dir: str,
    start_frame: int,
    end_frame: int,
    frame_step: int,
    max_pc_points: int,
):
    if not _SEQ_RE.match(sequence):
        raise ValueError(
            f"Invalid `--sequence` `{sequence}`. Expected format like `p000438_a000040`."
        )

    os.makedirs(save_dir, exist_ok=True)

    cameras_path = osp.join(data_root, "cameras", f"{sequence}_cameras.json")
    keypoints_path = osp.join(data_root, "skl", f"{sequence}_keypoints_3d.npz")
    smpl_path = osp.join(data_root, "smpl", f"{sequence}_smpl_params.npz")
    if not osp.exists(cameras_path):
        raise FileNotFoundError(f"Missing camera file: {cameras_path}")
    if not osp.exists(keypoints_path):
        raise FileNotFoundError(f"Missing GT keypoints file: {keypoints_path}")
    if not osp.exists(smpl_path):
        raise FileNotFoundError(f"Missing SMPL params file: {smpl_path}")

    with open(cameras_path, "r", encoding="utf-8") as f:
        cameras = json.load(f)
    gt_keypoints = np.load(keypoints_path)["keypoints_3d"].astype(np.float32)
    smpl = np.load(smpl_path)
    global_orient = smpl["global_orient"].astype(np.float32)
    if gt_keypoints.ndim != 3 or gt_keypoints.shape[-1] != 3:
        raise ValueError(f"Unexpected keypoints shape: {gt_keypoints.shape}")
    if global_orient.ndim != 2 or global_orient.shape[1] != 3:
        raise ValueError(f"Unexpected global_orient shape: {global_orient.shape}")

    total_frames = int(min(gt_keypoints.shape[0], global_orient.shape[0]))
    if total_frames <= 0:
        raise ValueError("No valid frame for canonical visualization.")
    if start_frame < 0:
        raise ValueError(f"`--start-frame` must be >= 0, got {start_frame}")
    if end_frame == -1:
        end_frame = total_frames - 1
    if end_frame < 0 or end_frame >= total_frames:
        raise ValueError(
            f"`--end-frame` out of range: {end_frame}. Valid range is [0, {total_frames - 1}] or -1."
        )
    if start_frame > end_frame:
        raise ValueError(f"`--start-frame` must be <= `--end-frame`, got {start_frame}>{end_frame}")
    if frame_step <= 0:
        raise ValueError(f"`--frame-step` must be > 0, got {frame_step}")

    rgb_views = _extract_camera_names(cameras, "rgb")
    depth_views = _extract_camera_names(cameras, "depth")
    if len(rgb_views) == 0:
        raise ValueError("No RGB camera views found in camera file.")
    if len(depth_views) == 0:
        raise ValueError("No depth camera views found in camera file.")

    frame_indices = list(range(start_frame, end_frame + 1, frame_step))
    seq_save_dir = osp.join(save_dir, sequence)
    os.makedirs(seq_save_dir, exist_ok=True)

    for frame_idx in tqdm(frame_indices, desc=f"visualize {sequence}", total=len(frame_indices)):
        kps_world = gt_keypoints[frame_idx]
        kps_canonical = _canonicalize_world_keypoints(
            keypoints_world=kps_world,
            global_orient=global_orient[frame_idx],
        )
        frame_dir = osp.join(seq_save_dir, f"frame_{frame_idx:06d}")
        os.makedirs(frame_dir, exist_ok=True)
        rgb_dir = osp.join(frame_dir, "rgb_views")
        depth_pc_dir = osp.join(frame_dir, "depth_pc_views")
        skl3d_dir = osp.join(frame_dir, "gt_skeleton_3d_views")
        skl2d_dir = osp.join(frame_dir, "gt_skeleton_2d_views")
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_pc_dir, exist_ok=True)
        os.makedirs(skl3d_dir, exist_ok=True)
        os.makedirs(skl2d_dir, exist_ok=True)

        for camera_name in rgb_views:
            _save_rgb_view(
                out_dir=rgb_dir,
                seq=sequence,
                frame_idx=frame_idx,
                data_root=data_root,
                camera_name=camera_name,
            )
            _save_skeleton_2d_view(
                out_dir=skl2d_dir,
                seq=sequence,
                frame_idx=frame_idx,
                data_root=data_root,
                keypoints_world=kps_world,
                camera_name=camera_name,
                cameras=cameras,
            )

        for camera_name in depth_views:
            _save_depth_pc_view(
                out_dir=depth_pc_dir,
                seq=sequence,
                frame_idx=frame_idx,
                data_root=data_root,
                camera_name=camera_name,
                cameras=cameras,
                max_pc_points=max_pc_points,
            )
            _save_skeleton_3d_view(
                out_dir=skl3d_dir,
                keypoints_world=kps_world,
                camera_name=camera_name,
                cameras=cameras,
            )

        _save_canonical_skeleton_3d(
            out_path=osp.join(frame_dir, "gt_skeleton_3d_canonical.png"),
            keypoints_canonical=kps_canonical,
        )

    print(f"Saved GT visualizations to: {seq_save_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize HUMMAN GT sequence across all camera views.")
    parser.add_argument("--sequence", type=str, required=True, help="Sequence name, e.g. p000438_a000040")
    parser.add_argument("--data-root", type=str, default="/root/autodl-tmp/humman_cropped")
    parser.add_argument("--save-dir", type=str, default="visualization_gt_sequence")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index (inclusive).")
    parser.add_argument("--end-frame", type=int, default=-1, help="End frame index (inclusive), -1 for last.")
    parser.add_argument("--frame-step", type=int, default=1, help="Frame step.")
    parser.add_argument("--max-pc-points", type=int, default=25000, help="Max depth-PC points per view.")
    return parser


def main():
    args = build_parser().parse_args()
    visualize_gt_sequence(
        data_root=args.data_root,
        sequence=args.sequence,
        save_dir=args.save_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        frame_step=args.frame_step,
        max_pc_points=args.max_pc_points,
    )


if __name__ == "__main__":
    main()
