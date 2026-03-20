from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh


@dataclass(frozen=True)
class VirtualLidarConfig:
    radius_range: tuple[float, float] = (2.5, 4.0)
    elevation_range_deg: tuple[float, float] = (5.0, 35.0)
    azimuth_range_deg: tuple[float, float] = (-180.0, 180.0)
    num_points: int = 2048
    oversample_factor: int = 8
    surface_noise_std: float = 0.002


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= 1e-8:
        raise ValueError("Cannot normalize near-zero vector.")
    return vec / norm


def look_at_world_to_sensor(sensor_position: np.ndarray, target: np.ndarray) -> np.ndarray:
    sensor_position = np.asarray(sensor_position, dtype=np.float32).reshape(3)
    target = np.asarray(target, dtype=np.float32).reshape(3)
    forward = _normalize(target - sensor_position)
    up_seed = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(forward, up_seed))) > 0.98:
        up_seed = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = _normalize(np.cross(up_seed, forward))
    up = _normalize(np.cross(forward, right))
    rotation = np.stack([right, up, forward], axis=0).astype(np.float32)
    translation = (-rotation @ sensor_position.reshape(3, 1)).astype(np.float32)
    return np.hstack([rotation, translation]).astype(np.float32)


def sample_virtual_lidar_pose(
    rng: random.Random,
    cfg: VirtualLidarConfig,
    target: np.ndarray | None = None,
) -> dict[str, Any]:
    if cfg.radius_range[0] <= 0 or cfg.radius_range[1] < cfg.radius_range[0]:
        raise ValueError(f"Invalid radius_range={cfg.radius_range}")
    target = np.zeros(3, dtype=np.float32) if target is None else np.asarray(target, dtype=np.float32).reshape(3)
    radius = rng.uniform(*cfg.radius_range)
    azimuth_deg = rng.uniform(*cfg.azimuth_range_deg)
    elevation_deg = rng.uniform(*cfg.elevation_range_deg)
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    x = radius * math.cos(elevation) * math.cos(azimuth)
    y = radius * math.sin(elevation)
    z = radius * math.cos(elevation) * math.sin(azimuth)
    sensor_position = np.array([x, y, z], dtype=np.float32) + target
    extrinsic = look_at_world_to_sensor(sensor_position=sensor_position, target=target)
    return {
        "sensor_position_world": sensor_position,
        "target_world": target.astype(np.float32),
        "extrinsic_world_to_sensor": extrinsic,
        "radius": radius,
        "azimuth_deg": azimuth_deg,
        "elevation_deg": elevation_deg,
    }


def sample_visible_surface_pointcloud(
    vertices_world: np.ndarray,
    faces: np.ndarray,
    *,
    sensor_position_world: np.ndarray,
    extrinsic_world_to_sensor: np.ndarray,
    cfg: VirtualLidarConfig,
    rng: random.Random,
) -> np.ndarray:
    mesh = trimesh.Trimesh(vertices=vertices_world, faces=faces, process=False)
    if mesh.faces.shape[0] == 0 or mesh.vertices.shape[0] == 0:
        raise ValueError("Mesh is empty; cannot synthesize point cloud.")

    desired = int(cfg.num_points)
    if desired <= 0:
        raise ValueError(f"num_points must be > 0, got {desired}")

    sample_count = desired * max(2, int(cfg.oversample_factor))
    points_world, face_idx = trimesh.sample.sample_surface(mesh, sample_count)
    normals = mesh.face_normals[face_idx]
    view_dirs = np.asarray(sensor_position_world, dtype=np.float32).reshape(1, 3) - points_world
    facing = np.sum(normals * view_dirs, axis=1) > 0.0
    points_world = points_world[facing]
    if points_world.shape[0] < desired:
        raise RuntimeError(
            f"Visible-surface sampling produced only {points_world.shape[0]} points; need {desired}."
        )

    chosen = rng.sample(range(points_world.shape[0]), desired)
    points_world = np.asarray(points_world[chosen], dtype=np.float32)
    rot = extrinsic_world_to_sensor[:, :3]
    trans = extrinsic_world_to_sensor[:, 3]
    points_sensor = (points_world @ rot.T) + trans[None, :]
    if cfg.surface_noise_std > 0.0:
        noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(
            loc=0.0,
            scale=cfg.surface_noise_std,
            size=points_sensor.shape,
        )
        points_sensor = points_sensor + noise.astype(np.float32)
    return points_sensor.astype(np.float32)
