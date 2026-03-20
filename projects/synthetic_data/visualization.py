from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_rgb_image(path: str | Path, image_rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"Failed to save image: {path}")


def save_mask_image(path: str | Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_uint8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if not cv2.imwrite(str(path), mask_uint8):
        raise RuntimeError(f"Failed to save mask: {path}")


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image_rgb = image_rgb.astype(np.uint8)
    mask = (mask > 0).astype(np.uint8)
    overlay = image_rgb.copy()
    tint = np.zeros_like(overlay)
    tint[..., 1] = 180
    blended = (0.65 * overlay + 0.35 * tint).astype(np.uint8)
    overlay[mask > 0] = blended[mask > 0]
    return overlay


@lru_cache(maxsize=1)
def _load_mhr70_edges() -> list[tuple[int, int, tuple[float, float, float]]]:
    repo_root = Path(__file__).resolve().parents[2]
    metadata_path = repo_root / "third_party" / "sam-3d-body" / "sam_3d_body" / "metadata" / "mhr70.py"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"MHR70 metadata file not found: {metadata_path}")

    spec = importlib.util.spec_from_file_location("sam3d_mhr70_metadata", metadata_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load import spec for MHR70 metadata: {metadata_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pose_info = module.pose_info
    keypoint_info = pose_info["keypoint_info"]
    name_to_idx = {entry["name"]: int(idx) for idx, entry in keypoint_info.items()}

    edges: list[tuple[int, int, tuple[float, float, float]]] = []
    for _, entry in sorted(pose_info["skeleton_info"].items()):
        start_name, end_name = entry["link"]
        color = tuple(float(c) / 255.0 for c in entry.get("color", [51, 153, 255]))
        if start_name not in name_to_idx or end_name not in name_to_idx:
            continue
        edges.append((name_to_idx[start_name], name_to_idx[end_name], color))
    return edges


def _plot_skeleton(ax, keypoints: np.ndarray, title: str) -> None:
    pts = np.asarray(keypoints, dtype=np.float32)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=8, c="tab:red")
    if pts.shape[0] == 70:
        edges = _load_mhr70_edges()
    else:
        edges = []
    for i, j, color in edges:
        if i >= pts.shape[0] or j >= pts.shape[0]:
            continue
        ax.plot(
            [pts[i, 0], pts[j, 0]],
            [pts[i, 1], pts[j, 1]],
            [pts[i, 2], pts[j, 2]],
            color=color,
            linewidth=0.8,
        )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=20, azim=35)


def _plot_sensor_context(ax, vertices_world: np.ndarray, sensor_position_world: np.ndarray) -> None:
    verts = np.asarray(vertices_world, dtype=np.float32)
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=0.5, c="lightgray")
    sensor = np.asarray(sensor_position_world, dtype=np.float32).reshape(3)
    ax.scatter([sensor[0]], [sensor[1]], [sensor[2]], s=40, c="tab:green")
    ax.plot([sensor[0], 0.0], [sensor[1], 0.0], [sensor[2], 0.0], color="tab:green", linewidth=1.0)
    ax.set_title("Virtual LiDAR")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=18, azim=30)


def _plot_pointcloud(ax, points_sensor: np.ndarray, title: str) -> None:
    pts = np.asarray(points_sensor, dtype=np.float32)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.0, c=pts[:, 2], cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=20, azim=55)


def save_summary_figure(
    path: str | Path,
    *,
    image_rgb: np.ndarray,
    mask: np.ndarray,
    reconstruction_overlay_rgb: np.ndarray,
    canonical_keypoints: np.ndarray,
    canonical_vertices: np.ndarray,
    sensor_position_world: np.ndarray,
    pointcloud_sensor: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 10))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(image_rgb)
    ax1.set_title("Source RGB")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(overlay_mask(image_rgb, mask))
    ax2.set_title("Saved Mask")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(reconstruction_overlay_rgb)
    ax3.set_title("SAM-3D-Body Overlay")
    ax3.axis("off")

    ax4 = fig.add_subplot(2, 3, 4, projection="3d")
    _plot_skeleton(ax4, canonical_keypoints, title="Canonical Keypoints (MHR70)")

    ax5 = fig.add_subplot(2, 3, 5, projection="3d")
    _plot_sensor_context(ax5, canonical_vertices, sensor_position_world)

    ax6 = fig.add_subplot(2, 3, 6, projection="3d")
    _plot_pointcloud(ax6, pointcloud_sensor, title="Synthetic LiDAR")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
