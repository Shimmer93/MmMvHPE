from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_rgb_cameras(rgb_camera: Any) -> list[dict[str, np.ndarray]]:
    """Normalize rgb_camera payload to a list of camera dicts."""
    if rgb_camera is None:
        return []
    if isinstance(rgb_camera, dict):
        cameras = [rgb_camera]
    elif isinstance(rgb_camera, (list, tuple)):
        cameras = list(rgb_camera)
    else:
        raise ValueError(f"Unsupported rgb_camera type: {type(rgb_camera)!r}")

    out: list[dict[str, np.ndarray]] = []
    for idx, cam in enumerate(cameras):
        if not isinstance(cam, dict):
            raise ValueError(f"rgb_camera[{idx}] must be a dict, got {type(cam)!r}")
        if "extrinsic" not in cam:
            raise ValueError(f"rgb_camera[{idx}] missing required key `extrinsic`.")
        extrinsic = _to_numpy(cam["extrinsic"]).astype(np.float32)
        if extrinsic.shape != (3, 4):
            raise ValueError(
                f"rgb_camera[{idx}].extrinsic must have shape (3, 4), got {extrinsic.shape}."
            )
        intrinsic = cam.get("intrinsic")
        intrinsic_np = None if intrinsic is None else _to_numpy(intrinsic).astype(np.float32)
        out.append({"intrinsic": intrinsic_np, "extrinsic": extrinsic})
    return out


def resolve_view_extrinsics(
    rgb_camera: Any,
    num_rgb_views: int,
) -> list[np.ndarray]:
    """Return one extrinsic per RGB view with strict validation."""
    cameras = normalize_rgb_cameras(rgb_camera)
    if not cameras:
        raise ValueError("Missing `rgb_camera` metadata in sample.")

    if len(cameras) == 1 and num_rgb_views >= 1:
        return [cameras[0]["extrinsic"] for _ in range(num_rgb_views)]

    if len(cameras) != num_rgb_views:
        raise ValueError(
            "Mismatch between RGB views and rgb_camera entries: "
            f"{num_rgb_views} view(s) vs {len(cameras)} camera(s)."
        )
    return [cam["extrinsic"] for cam in cameras]


def transform_points_to_camera(points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """Apply X_cam = R * X + t to points with shape (..., J, 3)."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[-1] != 3:
        raise ValueError(f"Expected points with last dim 3, got shape {pts.shape}.")

    ext = np.asarray(extrinsic, dtype=np.float32)
    if ext.shape != (3, 4):
        raise ValueError(f"Expected extrinsic with shape (3,4), got {ext.shape}.")

    rot = ext[:, :3]
    trans = ext[:, 3]
    flat = pts.reshape(-1, 3)
    cam = (flat @ rot.T) + trans[None, :]
    return cam.reshape(pts.shape)


def resolve_reference_extrinsic(
    sample: dict[str, Any],
    sensor_label: str,
    view_index: int,
) -> np.ndarray:
    """Resolve one reference extrinsic from sample camera payloads."""
    if view_index < 0:
        raise ValueError(f"reference view index must be >= 0, got {view_index}")

    label = str(sensor_label).lower().strip()
    if label == "rgb":
        key = "rgb_camera"
    elif label == "depth":
        key = "depth_camera"
    elif label == "lidar":
        # HuMMan uses depth-derived lidar; prefer lidar_camera and fall back to depth_camera.
        key = "lidar_camera" if sample.get("lidar_camera") is not None else "depth_camera"
    else:
        raise ValueError(
            f"Unsupported reference sensor `{sensor_label}`. "
            "Expected one of: rgb, depth, lidar."
        )

    cameras = normalize_rgb_cameras(sample.get(key))
    if not cameras:
        raise ValueError(
            f"Missing camera metadata for sensor `{label}` (expected `{key}` in sample)."
        )
    if view_index >= len(cameras):
        raise ValueError(
            f"Reference view index {view_index} out of range for `{label}`: "
            f"{len(cameras)} available view(s)."
        )
    return cameras[view_index]["extrinsic"]
