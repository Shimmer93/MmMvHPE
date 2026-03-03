from __future__ import annotations

import numpy as np


def rotate_points_y(points: np.ndarray, angle_rad: float) -> np.ndarray:
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rot = np.array(
        [[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]],
        dtype=np.float32,
    )
    return np.matmul(points, rot.T)


def rotate_points_x(points: np.ndarray, angle_rad: float) -> np.ndarray:
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rot = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_a, -sin_a], [0.0, sin_a, cos_a]],
        dtype=np.float32,
    )
    return np.matmul(points, rot.T)


def rotate_points_180_y(points: np.ndarray) -> np.ndarray:
    return rotate_points_y(points, np.deg2rad(180.0))


def align_keypoints_to_joints_scale(
    keypoints: np.ndarray,
    joints: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    pelvis_kpts = keypoints[0]
    pelvis_joints = joints[0]
    kpts_rel = keypoints - pelvis_kpts
    joints_rel = joints - pelvis_joints
    kpts_d = np.linalg.norm(kpts_rel[1:], axis=1).mean()
    joints_d = np.linalg.norm(joints_rel[1:], axis=1).mean()
    if kpts_d < eps or joints_d < eps:
        return keypoints
    scale = joints_d / kpts_d
    return kpts_rel * scale + pelvis_joints

