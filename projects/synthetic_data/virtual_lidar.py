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
    depth_buffer_resolution: int = 720
    depth_buffer_visibility_tolerance: float = 0.03
    depth_buffer_margin_ratio: float = 0.05
    depth_buffer_candidate_factor: int = 48


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


def _project_sensor_points_to_depth_map(
    points_sensor: np.ndarray,
    *,
    resolution: int,
    margin_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    pts = np.asarray(points_sensor, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points_sensor shape (N, 3), got {pts.shape}")
    if resolution <= 0:
        raise ValueError(f"depth_buffer_resolution must be > 0, got {resolution}")

    z = pts[:, 2]
    valid = z > 1e-6
    pts = pts[valid]
    if pts.shape[0] == 0:
        raise RuntimeError("No points remain in front of the sensor for depth-buffer projection.")

    xy_over_z = pts[:, :2] / pts[:, 2:3]
    scale = float(np.max(np.abs(xy_over_z)))
    scale = max(scale, 1e-3) * (1.0 + max(0.0, float(margin_ratio)))

    u = ((xy_over_z[:, 0] / scale) + 1.0) * 0.5 * float(resolution - 1)
    v = ((-xy_over_z[:, 1] / scale) + 1.0) * 0.5 * float(resolution - 1)

    in_bounds = (
        (u >= 0.0)
        & (u <= float(resolution - 1))
        & (v >= 0.0)
        & (v <= float(resolution - 1))
    )
    pts = pts[in_bounds]
    if pts.shape[0] == 0:
        raise RuntimeError("No points remain inside the projected depth-buffer bounds.")

    u = u[in_bounds]
    v = v[in_bounds]
    ix = np.rint(u).astype(np.int32)
    iy = np.rint(v).astype(np.int32)
    flat_idx = iy * resolution + ix
    return pts.astype(np.float32), ix, iy, flat_idx.astype(np.int64), float(scale)


def sample_self_occlusion_aware_pointcloud(
    vertices_world: np.ndarray,
    faces: np.ndarray,
    *,
    sensor_position_world: np.ndarray,
    extrinsic_world_to_sensor: np.ndarray,
    cfg: VirtualLidarConfig,
    rng: random.Random,
) -> tuple[np.ndarray, dict[str, Any]]:
    mesh = trimesh.Trimesh(vertices=vertices_world, faces=faces, process=False)
    if mesh.faces.shape[0] == 0 or mesh.vertices.shape[0] == 0:
        raise ValueError("Mesh is empty; cannot synthesize point cloud.")

    desired = int(cfg.num_points)
    if desired <= 0:
        raise ValueError(f"num_points must be > 0, got {desired}")
    if int(cfg.depth_buffer_resolution) <= 0:
        raise ValueError(
            f"depth_buffer_resolution must be > 0, got {cfg.depth_buffer_resolution}."
        )
    if float(cfg.depth_buffer_visibility_tolerance) < 0.0:
        raise ValueError(
            f"depth_buffer_visibility_tolerance must be >= 0, got {cfg.depth_buffer_visibility_tolerance}."
        )
    if int(cfg.depth_buffer_candidate_factor) <= 0:
        raise ValueError(
            f"depth_buffer_candidate_factor must be > 0, got {cfg.depth_buffer_candidate_factor}."
        )

    resolution = int(cfg.depth_buffer_resolution)
    base_candidate_count = max(
        desired * int(cfg.depth_buffer_candidate_factor),
        (resolution * resolution) // 8,
    )
    visibility_tolerance = float(cfg.depth_buffer_visibility_tolerance)
    margin_ratio = float(cfg.depth_buffer_margin_ratio)

    last_stats: dict[str, Any] | None = None
    sample_count = int(base_candidate_count)
    for attempt_idx in range(3):
        points_world, face_idx = trimesh.sample.sample_surface(mesh, sample_count)
        normals = mesh.face_normals[face_idx]
        view_dirs = np.asarray(sensor_position_world, dtype=np.float32).reshape(1, 3) - points_world
        facing = np.sum(normals * view_dirs, axis=1) > 0.0
        points_world = np.asarray(points_world[facing], dtype=np.float32)
        if points_world.shape[0] == 0:
            last_stats = {
                "candidate_count": int(sample_count),
                "after_facing_count": 0,
                "attempt_index": attempt_idx,
            }
            sample_count *= 2
            continue

        rot = np.asarray(extrinsic_world_to_sensor[:, :3], dtype=np.float32)
        trans = np.asarray(extrinsic_world_to_sensor[:, 3], dtype=np.float32)
        points_sensor = (points_world @ rot.T) + trans[None, :]

        try:
            projected_points_sensor, _, _, flat_idx, projection_scale = _project_sensor_points_to_depth_map(
                points_sensor,
                resolution=resolution,
                margin_ratio=margin_ratio,
            )
        except RuntimeError:
            last_stats = {
                "candidate_count": int(sample_count),
                "after_facing_count": int(points_world.shape[0]),
                "after_projection_count": 0,
                "attempt_index": attempt_idx,
            }
            sample_count *= 2
            continue

        z = projected_points_sensor[:, 2].astype(np.float32)
        depth_buffer = np.full((resolution * resolution,), np.inf, dtype=np.float32)
        np.minimum.at(depth_buffer, flat_idx, z)
        nearest_depth = depth_buffer[flat_idx]
        visible = z <= (nearest_depth + visibility_tolerance)
        visible_points_sensor = projected_points_sensor[visible]

        last_stats = {
            "candidate_count": int(sample_count),
            "after_facing_count": int(points_world.shape[0]),
            "after_projection_count": int(projected_points_sensor.shape[0]),
            "visible_count": int(visible_points_sensor.shape[0]),
            "depth_pixels_occupied": int(np.count_nonzero(np.isfinite(depth_buffer))),
            "projection_scale": float(projection_scale),
            "attempt_index": int(attempt_idx),
        }

        if visible_points_sensor.shape[0] < desired:
            sample_count *= 2
            continue

        chosen = rng.sample(range(visible_points_sensor.shape[0]), desired)
        points_sensor_out = np.asarray(visible_points_sensor[chosen], dtype=np.float32)
        if cfg.surface_noise_std > 0.0:
            noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(
                loc=0.0,
                scale=cfg.surface_noise_std,
                size=points_sensor_out.shape,
            )
            points_sensor_out = points_sensor_out + noise.astype(np.float32)

        metadata = {
            "simulation_mode": "depth_buffer_self_occlusion_v1",
            "depth_buffer_resolution": int(resolution),
            "depth_buffer_visibility_tolerance": float(visibility_tolerance),
            "depth_buffer_margin_ratio": float(margin_ratio),
            "depth_buffer_candidate_factor": int(cfg.depth_buffer_candidate_factor),
            "num_points": int(desired),
            "surface_noise_std": float(cfg.surface_noise_std),
            **last_stats,
        }
        return points_sensor_out.astype(np.float32), metadata

    raise RuntimeError(
        "Self-occlusion-aware sampling produced too few visible points "
        f"after 3 attempts. Last stats: {last_stats}"
    )
