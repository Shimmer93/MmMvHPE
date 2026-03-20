#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.synthetic_data.visualization import _load_mhr70_edges, overlay_mask  # noqa: E402


SMPL24_EDGES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 9),
    (7, 10),
    (8, 11),
    (9, 12),
    (12, 13),
    (12, 14),
    (12, 15),
    (13, 16),
    (14, 17),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21),
    (20, 22),
    (21, 23),
]

PANOPTIC19_EDGES = [
    (0, 1),
    (1, 15),
    (15, 16),
    (1, 17),
    (17, 18),
    (0, 2),
    (0, 3),
    (3, 4),
    (4, 5),
    (0, 9),
    (9, 10),
    (10, 11),
    (2, 6),
    (6, 7),
    (7, 8),
    (2, 12),
    (12, 13),
    (13, 14),
]

LEFT_COLOR = (30, 144, 255)
RIGHT_COLOR = (255, 99, 71)
CENTER_COLOR = (255, 215, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render QC figures for synthetic target-format exports.")
    parser.add_argument("--synthetic-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--sample-dir", action="append", default=None, help="Explicit sample directory to visualize.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument(
        "--selection",
        type=str,
        default="first",
        choices=["first", "spread"],
        help="How to select samples when --sample-dir is not provided.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_image_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    return mask


def denormalize_points_2d(points_norm: np.ndarray, image_size_hw: tuple[int, int]) -> np.ndarray:
    H, W = int(image_size_hw[0]), int(image_size_hw[1])
    pts = np.asarray(points_norm, dtype=np.float32).copy()
    pts[:, 0] = 0.5 * (pts[:, 0] + 1.0) * max(W - 1, 1)
    pts[:, 1] = 0.5 * (pts[:, 1] + 1.0) * max(H - 1, 1)
    return pts


def edge_color_for_joint_pair(i: int, j: int, joint_count: int) -> tuple[int, int, int]:
    if joint_count == 24:
        left = {1, 4, 7, 10, 13, 16, 18, 20, 22}
        right = {2, 5, 8, 11, 14, 17, 19, 21, 23}
    elif joint_count == 19:
        left = {3, 4, 5, 6, 7, 8, 15, 16}
        right = {9, 10, 11, 12, 13, 14, 17, 18}
    else:
        left = set()
        right = set()
    if i in left or j in left:
        return LEFT_COLOR
    if i in right or j in right:
        return RIGHT_COLOR
    return CENTER_COLOR


def _iter_edge_specs(edges: list[tuple], joint_count: int) -> list[tuple[int, int, np.ndarray]]:
    parsed = []
    for edge in edges:
        if len(edge) == 2:
            i, j = edge
            color = np.asarray(edge_color_for_joint_pair(i, j, joint_count), dtype=np.float32) / 255.0
        elif len(edge) == 3:
            i, j, color = edge
            color = np.asarray(color, dtype=np.float32)
            if color.max(initial=0.0) > 1.0:
                color = color / 255.0
        else:
            raise ValueError(f"Unsupported edge spec: {edge}")
        parsed.append((int(i), int(j), color.astype(np.float32)))
    return parsed


def draw_skeleton_2d(image_rgb: np.ndarray, points_2d: np.ndarray, edges: list[tuple[int, int]]) -> np.ndarray:
    canvas = image_rgb.copy()
    pts = np.asarray(points_2d, dtype=np.float32)
    for i, j in edges:
        if i >= pts.shape[0] or j >= pts.shape[0]:
            continue
        p0 = tuple(int(round(v)) for v in pts[i])
        p1 = tuple(int(round(v)) for v in pts[j])
        cv2.line(canvas, p0, p1, edge_color_for_joint_pair(i, j, pts.shape[0]), thickness=2, lineType=cv2.LINE_AA)
    for idx, point in enumerate(pts):
        color = edge_color_for_joint_pair(idx, idx, pts.shape[0])
        cv2.circle(canvas, tuple(int(round(v)) for v in point), radius=3, color=color, thickness=-1, lineType=cv2.LINE_AA)
    return canvas


def draw_bbox(image_rgb: np.ndarray, bbox_xywh: list[float]) -> np.ndarray:
    canvas = image_rgb.copy()
    x, y, w, h = [int(round(v)) for v in bbox_xywh]
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CENTER_COLOR, thickness=2, lineType=cv2.LINE_AA)
    return canvas


def sensor_points_to_world(points_sensor: np.ndarray, extrinsic_world_to_sensor: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_sensor, dtype=np.float32)
    extrinsic = np.asarray(extrinsic_world_to_sensor, dtype=np.float32).reshape(3, 4)
    R = extrinsic[:, :3]
    t = extrinsic[:, 3]
    return ((R.T @ (pts - t.reshape(1, 3)).T).T).astype(np.float32)


def set_axes_equal(ax, points: np.ndarray) -> None:
    pts = np.asarray(points, dtype=np.float32)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_skeleton_3d(ax, points: np.ndarray, edges: list[tuple], title: str) -> None:
    pts = np.asarray(points, dtype=np.float32)
    edge_specs = _iter_edge_specs(edges, pts.shape[0])
    for i, j, color in edge_specs:
        if i >= pts.shape[0] or j >= pts.shape[0]:
            continue
        ax.plot(
            [pts[i, 0], pts[j, 0]],
            [pts[i, 1], pts[j, 1]],
            [pts[i, 2], pts[j, 2]],
            color=color,
            linewidth=1.4,
        )
    colors = np.asarray([edge_color_for_joint_pair(i, i, pts.shape[0]) for i in range(pts.shape[0])], dtype=np.float32) / 255.0
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=16, c=colors)
    set_axes_equal(ax, pts)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=20, azim=35)


def plot_pointcloud_with_keypoints(ax, points: np.ndarray, keypoints: np.ndarray, edges: list[tuple[int, int]], title: str) -> None:
    pc = np.asarray(points, dtype=np.float32)
    kp = np.asarray(keypoints, dtype=np.float32)
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=0.8, c=pc[:, 2], cmap="viridis", alpha=0.7)
    edge_specs = _iter_edge_specs(edges, kp.shape[0])
    for i, j, color in edge_specs:
        if i >= kp.shape[0] or j >= kp.shape[0]:
            continue
        ax.plot([kp[i, 0], kp[j, 0]], [kp[i, 1], kp[j, 1]], [kp[i, 2], kp[j, 2]], color=color, linewidth=1.5)
    colors = np.asarray([edge_color_for_joint_pair(i, i, kp.shape[0]) for i in range(kp.shape[0])], dtype=np.float32) / 255.0
    ax.scatter(kp[:, 0], kp[:, 1], kp[:, 2], s=18, c=colors)
    set_axes_equal(ax, np.concatenate([pc[:, :3], kp], axis=0))
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=20, azim=55)


def render_text_panel(ax, sample_name: str, base_manifest: dict[str, Any], export_manifest: dict[str, Any]) -> None:
    humman = export_manifest["formats"]["humman"]
    conversion = humman.get("conversion", {})
    bbox = base_manifest.get("bbox_xywh", [0, 0, 0, 0])
    text = "\n".join(
        [
            sample_name,
            f"annotation_id: {base_manifest.get('annotation_id')}",
            f"image_id: {base_manifest.get('image_id')}",
            f"image_hw: {tuple(base_manifest.get('image_size_hw', []))}",
            f"bbox_xywh: {[round(float(v), 1) for v in bbox]}",
            f"mask: {base_manifest.get('mask_provenance')}",
            f"sam3d_mask_assisted: {base_manifest.get('mask_assisted_sam3d')}",
            f"backend: {conversion.get('selected_backend')}",
            f"fit_err: {conversion.get('fitting_error', float('nan')):.5f}",
            f"edge_err: {conversion.get('edge_error', float('nan')):.5f}",
            f"lidar_radius: {base_manifest.get('lidar_pose', {}).get('radius', float('nan')):.2f}",
            f"lidar_azim: {base_manifest.get('lidar_pose', {}).get('azimuth_deg', float('nan')):.1f}",
            f"lidar_elev: {base_manifest.get('lidar_pose', {}).get('elevation_deg', float('nan')):.1f}",
        ]
    )
    ax.axis("off")
    ax.text(0.02, 0.98, text, va="top", ha="left", family="monospace", fontsize=10)


def render_sample(sample_dir: Path, output_path: Path) -> None:
    base_manifest = read_json(sample_dir / "manifest.json")
    export_manifest = read_json(sample_dir / "exports" / "export_manifest.json")
    humman_manifest = read_json(Path(export_manifest["formats"]["humman"]["manifest_path"]))
    panoptic_manifest = read_json(Path(export_manifest["formats"]["panoptic"]["manifest_path"]))

    image = load_image_rgb(Path(base_manifest["image_path"]))
    mask = load_mask(Path(base_manifest["artifacts"]["source_mask"]))
    image_hw = tuple(int(v) for v in base_manifest["image_size_hw"])
    mhr_edges = _load_mhr70_edges()

    humman_kp_2d = denormalize_points_2d(
        np.load(humman_manifest["artifacts"]["gt_keypoints_2d_rgb"]).astype(np.float32),
        image_hw,
    )
    panoptic_kp_2d = denormalize_points_2d(
        np.load(panoptic_manifest["artifacts"]["gt_keypoints_2d_rgb"]).astype(np.float32),
        image_hw,
    )
    mhr_kp_world = np.load(humman_manifest["artifacts"]["mhr_keypoints_world"]).astype(np.float32)
    humman_kp_3d = np.load(humman_manifest["artifacts"]["gt_keypoints_world"]).astype(np.float32)
    panoptic_kp_3d = np.load(panoptic_manifest["artifacts"]["gt_keypoints_world"]).astype(np.float32)
    input_lidar = np.load(humman_manifest["artifacts"]["input_lidar"]).astype(np.float32)
    lidar_extrinsic_canonical = np.load(base_manifest["artifacts"]["lidar_extrinsic_world_to_sensor"]).astype(np.float32)
    canonical_pointcloud = sensor_points_to_world(input_lidar, lidar_extrinsic_canonical)
    mhr_pelvis_world = 0.5 * (mhr_kp_world[9] + mhr_kp_world[10])
    pointcloud_world = canonical_pointcloud + mhr_pelvis_world.reshape(1, 3)

    fig = plt.figure(figsize=(22, 15))

    ax1 = fig.add_subplot(3, 4, 1)
    ax1.imshow(image)
    ax1.set_title("Source RGB")
    ax1.axis("off")

    ax2 = fig.add_subplot(3, 4, 2)
    ax2.imshow(draw_bbox(overlay_mask(image, mask), base_manifest["bbox_xywh"]))
    ax2.set_title("Mask + BBox")
    ax2.axis("off")

    ax3 = fig.add_subplot(3, 4, 3)
    ax3.imshow(draw_skeleton_2d(image, humman_kp_2d, SMPL24_EDGES))
    ax3.set_title("HuMMan 2D Projection")
    ax3.axis("off")

    ax4 = fig.add_subplot(3, 4, 4)
    ax4.imshow(draw_skeleton_2d(image, panoptic_kp_2d, PANOPTIC19_EDGES))
    ax4.set_title("Panoptic 2D Projection")
    ax4.axis("off")

    ax5 = fig.add_subplot(3, 4, 5, projection="3d")
    plot_skeleton_3d(ax5, mhr_kp_world, mhr_edges, "MHR70 World")

    ax6 = fig.add_subplot(3, 4, 6, projection="3d")
    plot_skeleton_3d(ax6, humman_kp_3d, SMPL24_EDGES, "HuMMan SMPL24 World")

    ax7 = fig.add_subplot(3, 4, 7, projection="3d")
    plot_skeleton_3d(ax7, panoptic_kp_3d, PANOPTIC19_EDGES, "Panoptic19 World")

    ax8 = fig.add_subplot(3, 4, 8, projection="3d")
    plot_pointcloud_with_keypoints(ax8, pointcloud_world, mhr_kp_world, mhr_edges, "World PointCloud + MHR70")

    ax9 = fig.add_subplot(3, 4, 9, projection="3d")
    plot_pointcloud_with_keypoints(ax9, pointcloud_world, humman_kp_3d, SMPL24_EDGES, "World PointCloud + SMPL24")

    ax10 = fig.add_subplot(3, 4, 10, projection="3d")
    plot_pointcloud_with_keypoints(ax10, pointcloud_world, panoptic_kp_3d, PANOPTIC19_EDGES, "World PointCloud + Panoptic19")

    ax11 = fig.add_subplot(3, 4, 11)
    ax11.imshow(draw_skeleton_2d(draw_bbox(overlay_mask(image, mask), base_manifest["bbox_xywh"]), humman_kp_2d, SMPL24_EDGES))
    ax11.set_title("Mask + HuMMan 2D")
    ax11.axis("off")

    ax12 = fig.add_subplot(3, 4, 12)
    render_text_panel(ax12, sample_dir.name, base_manifest, export_manifest)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def collect_exported_sample_dirs(synthetic_root: Path) -> list[Path]:
    sample_dirs = []
    for path in sorted(synthetic_root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "exports" / "export_manifest.json").is_file():
            sample_dirs.append(path)
    return sample_dirs


def select_sample_dirs(sample_dirs: list[Path], start_index: int, max_samples: int, selection: str) -> list[Path]:
    if max_samples <= 0:
        return []
    if selection == "first" or len(sample_dirs) <= max_samples:
        return sample_dirs[start_index : start_index + max_samples]
    available = sample_dirs[start_index:]
    if len(available) <= max_samples:
        return available
    positions = np.linspace(0, len(available) - 1, num=max_samples)
    chosen = []
    seen = set()
    for pos in positions:
        idx = int(round(float(pos)))
        idx = min(max(idx, 0), len(available) - 1)
        if idx in seen:
            continue
        seen.add(idx)
        chosen.append(available[idx])
    return chosen


def build_contact_sheet(image_paths: list[Path], output_path: Path, num_cols: int = 2) -> None:
    if not image_paths:
        raise ValueError("No image paths provided for contact sheet.")
    images = []
    tile_w = 1000
    tile_h = 500
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to load rendered QC image: {path}")
        image = cv2.resize(image, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        images.append(image)
    num_cols = max(1, int(num_cols))
    num_rows = int(math.ceil(len(images) / num_cols))
    canvas = np.full((num_rows * tile_h, num_cols * tile_w, 3), 245, dtype=np.uint8)
    for idx, image in enumerate(images):
        row = idx // num_cols
        col = idx % num_cols
        y0 = row * tile_h
        x0 = col * tile_w
        canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Failed to save contact sheet: {output_path}")


def main() -> None:
    args = parse_args()
    synthetic_root = Path(args.synthetic_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not synthetic_root.is_dir():
        raise FileNotFoundError(f"Synthetic root not found: {synthetic_root}")

    if args.sample_dir:
        sample_dirs = [Path(path).expanduser().resolve() for path in args.sample_dir]
    else:
        exported = collect_exported_sample_dirs(synthetic_root)
        if not exported:
            raise FileNotFoundError(
                f"No exported samples found under {synthetic_root}. Run scripts/export_synthetic_gt_formats.py first."
            )
        sample_dirs = select_sample_dirs(exported, args.start_index, args.max_samples, args.selection)

    if not sample_dirs:
        raise ValueError("No sample directories selected.")

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_paths: list[Path] = []
    for sample_dir in sample_dirs:
        output_path = output_dir / f"{sample_dir.name}.png"
        render_sample(sample_dir, output_path)
        rendered_paths.append(output_path)
        print(f"rendered: {output_path}")

    build_contact_sheet(rendered_paths, output_dir / "index.png", num_cols=min(2, len(rendered_paths)))
    summary = {
        "synthetic_root": str(synthetic_root),
        "output_dir": str(output_dir),
        "num_samples": len(sample_dirs),
        "samples": [str(path) for path in sample_dirs],
        "index_image": str(output_dir / "index.png"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
