from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from misc.pose_enc import extri_intri_to_pose_encoding

MHR70_NECK_IDX = 69
MHR70_NOSE_IDX = 0
MHR70_LEFT_EYE_IDX = 1
MHR70_RIGHT_EYE_IDX = 2
MHR70_LEFT_EAR_IDX = 3
MHR70_RIGHT_EAR_IDX = 4
MHR70_LEFT_SHOULDER_IDX = 5
MHR70_RIGHT_SHOULDER_IDX = 6
MHR70_LEFT_ELBOW_IDX = 7
MHR70_RIGHT_ELBOW_IDX = 8
MHR70_LEFT_HIP_IDX = 9
MHR70_RIGHT_HIP_IDX = 10
MHR70_LEFT_KNEE_IDX = 11
MHR70_RIGHT_KNEE_IDX = 12
MHR70_LEFT_ANKLE_IDX = 13
MHR70_RIGHT_ANKLE_IDX = 14
MHR70_RIGHT_WRIST_IDX = 41
MHR70_LEFT_WRIST_IDX = 62


SMPL24_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]

PANOPTIC19_JOINT_NAMES = [
    "neck",
    "nose",
    "body_center",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_eye",
    "left_ear",
    "right_eye",
    "right_ear",
]


@dataclass(frozen=True)
class TopologyMetadata:
    name: str
    joint_names: tuple[str, ...]
    root_joint_indices: tuple[int, ...]
    coordinate_space: str


TOPOLOGY_METADATA = {
    "smpl24_new_world": TopologyMetadata(
        name="smpl24_new_world",
        joint_names=tuple(SMPL24_JOINT_NAMES),
        root_joint_indices=(0,),
        coordinate_space="new_world",
    ),
    "smpl24_world": TopologyMetadata(
        name="smpl24_world",
        joint_names=tuple(SMPL24_JOINT_NAMES),
        root_joint_indices=(0,),
        coordinate_space="world",
    ),
    "panoptic19_new_world": TopologyMetadata(
        name="panoptic19_new_world",
        joint_names=tuple(PANOPTIC19_JOINT_NAMES),
        root_joint_indices=(2,),
        coordinate_space="new_world",
    ),
    "panoptic19_world": TopologyMetadata(
        name="panoptic19_world",
        joint_names=tuple(PANOPTIC19_JOINT_NAMES),
        root_joint_indices=(2,),
        coordinate_space="world",
    ),
}


def axis_angle_to_matrix_np(axis_angle: np.ndarray) -> np.ndarray:
    axis_angle = np.asarray(axis_angle, dtype=np.float32).reshape(3)
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = axis_angle / angle
    x, y, z = axis
    K = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float32,
    )
    eye = np.eye(3, dtype=np.float32)
    return eye + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def normalize_vec(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n <= eps:
        raise ValueError("Cannot normalize near-zero vector.")
    return (v / n).astype(np.float32)


def estimate_panoptic_root_rotation(joints19_world: np.ndarray) -> np.ndarray:
    joints19_world = np.asarray(joints19_world, dtype=np.float32)
    if joints19_world.shape != (19, 3):
        raise ValueError(f"Expected Panoptic joints19 shape (19,3), got {joints19_world.shape}")
    neck = joints19_world[0]
    body = joints19_world[2]
    left_hip = joints19_world[6]
    right_hip = joints19_world[12]
    x_axis = normalize_vec(right_hip - left_hip)
    y_seed = normalize_vec(neck - body)
    z_axis = normalize_vec(np.cross(x_axis, y_seed))
    y_axis = normalize_vec(np.cross(z_axis, x_axis))
    R_new_to_world = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)
    det = float(np.linalg.det(R_new_to_world))
    if not np.isfinite(det) or abs(det) < 1e-5:
        raise ValueError(f"Invalid Panoptic root rotation matrix (det={det}).")
    return R_new_to_world


def world_to_new_world(points_world: np.ndarray, pelvis_world: np.ndarray, R_new_to_world: np.ndarray) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float32)
    pelvis_world = np.asarray(pelvis_world, dtype=np.float32).reshape(3)
    R_new_to_world = np.asarray(R_new_to_world, dtype=np.float32).reshape(3, 3)
    return (R_new_to_world.T @ (points_world - pelvis_world).T).T.astype(np.float32)


def update_extrinsic_for_new_world(
    extrinsic_world_to_camera: np.ndarray,
    pelvis_world: np.ndarray,
    R_new_to_world: np.ndarray,
) -> np.ndarray:
    extrinsic_world_to_camera = np.asarray(extrinsic_world_to_camera, dtype=np.float32).reshape(3, 4)
    pelvis_world = np.asarray(pelvis_world, dtype=np.float32).reshape(3)
    R_wc = extrinsic_world_to_camera[:, :3]
    T_wc = extrinsic_world_to_camera[:, 3:4]
    R_new = R_wc @ np.asarray(R_new_to_world, dtype=np.float32).reshape(3, 3)
    T_new = R_wc @ pelvis_world.reshape(3, 1) + T_wc
    return np.hstack([R_new, T_new]).astype(np.float32)


def transform_points_to_camera(points_world: np.ndarray, extrinsic_world_to_camera: np.ndarray) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float32)
    extrinsic_world_to_camera = np.asarray(extrinsic_world_to_camera, dtype=np.float32).reshape(3, 4)
    R = extrinsic_world_to_camera[:, :3]
    T = extrinsic_world_to_camera[:, 3]
    return ((R @ points_world.T).T + T.reshape(1, 3)).astype(np.float32)


def project_points_to_image(points_world: np.ndarray, camera: dict[str, Any]) -> np.ndarray:
    intrinsic = np.asarray(camera["intrinsic"], dtype=np.float32).reshape(3, 3)
    extrinsic = np.asarray(camera["extrinsic"], dtype=np.float32).reshape(3, 4)
    points_cam = transform_points_to_camera(points_world, extrinsic)
    z = np.clip(points_cam[:, 2], 1e-6, None)
    x = intrinsic[0, 0] * (points_cam[:, 0] / z) + intrinsic[0, 2]
    y = intrinsic[1, 1] * (points_cam[:, 1] / z) + intrinsic[1, 2]
    return np.stack([x, y], axis=-1).astype(np.float32)


def normalize_points_2d(points_2d: np.ndarray, image_size_hw: tuple[int, int]) -> np.ndarray:
    H, W = int(image_size_hw[0]), int(image_size_hw[1])
    out = np.asarray(points_2d, dtype=np.float32).copy()
    out[:, 0] = 2.0 * (out[:, 0] / max(W - 1, 1)) - 1.0
    out[:, 1] = 2.0 * (out[:, 1] / max(H - 1, 1)) - 1.0
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def build_rgb_camera_from_sam3d(
    image_size_hw: tuple[int, int],
    focal_length: float | np.ndarray,
    extrinsic_world_to_camera: np.ndarray,
) -> dict[str, np.ndarray]:
    H, W = int(image_size_hw[0]), int(image_size_hw[1])
    if np.ndim(focal_length) == 0:
        fx = fy = float(focal_length)
    else:
        focal = np.asarray(focal_length, dtype=np.float32).reshape(-1)
        if focal.size == 1:
            fx = fy = float(focal[0])
        else:
            fx = float(focal[0])
            fy = float(focal[1])
    intrinsic = np.array(
        [
            [fx, 0.0, W / 2.0],
            [0.0, fy, H / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    extrinsic = np.asarray(extrinsic_world_to_camera, dtype=np.float32).reshape(3, 4)
    return {"intrinsic": intrinsic, "extrinsic": extrinsic}


def camera_to_pose_encoding(camera: dict[str, Any], image_size_hw: tuple[int, int]) -> np.ndarray:
    extrinsic = torch.from_numpy(np.asarray(camera["extrinsic"], dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    intrinsic = torch.from_numpy(np.asarray(camera["intrinsic"], dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    pose = extri_intri_to_pose_encoding(
        extrinsic,
        intrinsic,
        image_size_hw=tuple(int(x) for x in image_size_hw),
        pose_encoding_type="absT_quaR_FoV",
    )
    return pose.squeeze(0).cpu().numpy().astype(np.float32)


def center_keypoints_with_pc(keypoints_lidar: np.ndarray, input_lidar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keypoints_lidar = np.asarray(keypoints_lidar, dtype=np.float32)
    input_lidar = np.asarray(input_lidar, dtype=np.float32)
    if input_lidar.ndim != 2 or input_lidar.shape[1] < 3:
        raise ValueError(f"Expected input_lidar shape (N,C>=3), got {input_lidar.shape}")
    center = np.mean(input_lidar[:, :3], axis=0).astype(np.float32)
    centered = (keypoints_lidar - center.reshape(1, 3)).astype(np.float32)
    return centered, center


def mhr70_to_panoptic19(mhr70_keypoints: np.ndarray) -> np.ndarray:
    mhr70_keypoints = np.asarray(mhr70_keypoints, dtype=np.float32)
    if mhr70_keypoints.shape != (70, 3):
        raise ValueError(f"Expected MHR70 keypoints shape (70,3), got {mhr70_keypoints.shape}")
    body_center = 0.5 * (mhr70_keypoints[MHR70_LEFT_HIP_IDX] + mhr70_keypoints[MHR70_RIGHT_HIP_IDX])
    out = np.stack(
        [
            mhr70_keypoints[MHR70_NECK_IDX],
            mhr70_keypoints[MHR70_NOSE_IDX],
            body_center,
            mhr70_keypoints[MHR70_LEFT_SHOULDER_IDX],
            mhr70_keypoints[MHR70_LEFT_ELBOW_IDX],
            mhr70_keypoints[MHR70_LEFT_WRIST_IDX],
            mhr70_keypoints[MHR70_LEFT_HIP_IDX],
            mhr70_keypoints[MHR70_LEFT_KNEE_IDX],
            mhr70_keypoints[MHR70_LEFT_ANKLE_IDX],
            mhr70_keypoints[MHR70_RIGHT_SHOULDER_IDX],
            mhr70_keypoints[MHR70_RIGHT_ELBOW_IDX],
            mhr70_keypoints[MHR70_RIGHT_WRIST_IDX],
            mhr70_keypoints[MHR70_RIGHT_HIP_IDX],
            mhr70_keypoints[MHR70_RIGHT_KNEE_IDX],
            mhr70_keypoints[MHR70_RIGHT_ANKLE_IDX],
            mhr70_keypoints[MHR70_LEFT_EYE_IDX],
            mhr70_keypoints[MHR70_LEFT_EAR_IDX],
            mhr70_keypoints[MHR70_RIGHT_EYE_IDX],
            mhr70_keypoints[MHR70_RIGHT_EAR_IDX],
        ],
        axis=0,
    )
    return out.astype(np.float32)


def ensure_relative_symlink(link_path: Path, target_path: Path) -> None:
    link_path = Path(link_path)
    target_path = Path(target_path)
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    try:
        relative_target = Path(os.path.relpath(target_path, start=link_path.parent))
    except Exception:
        relative_target = target_path
    link_path.symlink_to(relative_target)


def save_camera_json(path: Path, camera: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "intrinsic": np.asarray(camera["intrinsic"], dtype=np.float32).astype(float).tolist(),
        "extrinsic": np.asarray(camera["extrinsic"], dtype=np.float32).astype(float).tolist(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(array))
