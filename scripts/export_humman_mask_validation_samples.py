#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import HummanPreprocessedDatasetV2
from datasets.transforms.humman_mask_transforms import (
    load_humman_binary_mask,
    resolve_humman_mask_path,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export HuMMan runtime-mask validation samples with RGB overlays, "
            "masked depth previews, and masked/unmasked LiDAR point clouds."
        )
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/opt/data/humman_cropped"),
        help="Root of the preprocessed HuMMan dataset.",
    )
    p.add_argument(
        "--mask-root",
        type=Path,
        default=Path("/opt/data/humman_cropped_masks"),
        help="Root of the flat HuMMan mask files.",
    )
    p.add_argument("--sequence", type=str, required=True, help="Sequence name, e.g. p000438_a000040.")
    p.add_argument(
        "--cameras",
        type=str,
        default="kinect_000",
        help="Comma-separated camera names, e.g. kinect_000,kinect_001.",
    )
    p.add_argument(
        "--frame-ids",
        type=str,
        default=None,
        help="Optional comma-separated 6-digit frame ids to export for every camera.",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of frames to export per camera when --frame-ids is omitted.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("logs/humman_mask_validation"),
        help="Output root for exported artifacts.",
    )
    return p.parse_args()


def _write_ply_ascii(path: Path, points_xyz: np.ndarray) -> None:
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"points_xyz must be (N,>=3), got {pts.shape}")
    pts = pts[:, :3]
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f}\n")


def _depth_preview(depth: np.ndarray) -> np.ndarray:
    depth_f = np.asarray(depth, dtype=np.float32)
    nz = depth_f[depth_f > 0]
    if nz.size == 0:
        norm = np.zeros_like(depth_f, dtype=np.float32)
    else:
        lo, hi = np.percentile(nz, [2.0, 98.0]).astype(np.float32)
        norm = np.clip((depth_f - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    depth_u8 = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)


def _mask_overlay(rgb: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = rgb_bgr.copy()
    green = np.zeros_like(overlay)
    green[..., 1] = 255
    overlay[mask_bool] = cv2.addWeighted(overlay[mask_bool], 0.5, green[mask_bool], 0.5, 0.0)
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, contourIdx=-1, color=(0, 0, 255), thickness=1)
    return overlay


def _sample_pointcloud(points: np.ndarray, max_points: int = 4000) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] <= max_points:
        return pts
    idx = np.linspace(0, pts.shape[0] - 1, num=max_points, dtype=np.int64)
    return pts[idx]


def _plot_pointcloud(ax: plt.Axes, points: np.ndarray, title: str) -> None:
    pts = _sample_pointcloud(points)
    ax.set_title(title)
    if pts.size == 0:
        ax.text(0.5, 0.5, "empty", ha="center", va="center")
        ax.set_axis_off()
        return
    sc = ax.scatter(pts[:, 0], pts[:, 2], c=pts[:, 1], s=1.0, cmap="viridis", linewidths=0)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)


def _normalize_cameras(cameras_arg: str) -> List[str]:
    cameras = [x.strip() for x in str(cameras_arg).split(",") if x.strip()]
    if not cameras:
        raise ValueError("At least one camera must be provided.")
    return cameras


def _parse_frame_ids(frame_ids_arg: str | None) -> List[int] | None:
    if frame_ids_arg is None:
        return None
    out = [int(x.strip()) for x in frame_ids_arg.split(",") if x.strip()]
    if not out:
        raise ValueError("--frame-ids was provided but no valid frame ids were parsed.")
    return out


def _build_datasets(data_root: Path, mask_root: Path, camera: str) -> Tuple[HummanPreprocessedDatasetV2, HummanPreprocessedDatasetV2]:
    common_params = dict(
        data_root=str(data_root),
        split="all",
        modality_names=("rgb", "depth", "lidar"),
        rgb_cameras=[camera],
        depth_cameras=[camera],
        rgb_cameras_per_sample=1,
        depth_cameras_per_sample=1,
        lidar_cameras_per_sample=1,
        seq_len=1,
        seq_step=1,
        convert_depth_to_lidar=False,
    )
    plain_ds = HummanPreprocessedDatasetV2(pipeline=[], **common_params)
    masked_ds = HummanPreprocessedDatasetV2(
        pipeline=[
            {
                "name": "ApplyHummanSegmentationMask",
                "params": {
                    "apply_to": ["rgb", "depth", "lidar"],
                    "mask_root": str(mask_root),
                },
            }
        ],
        **common_params,
    )
    return plain_ds, masked_ds


def _sequence_frame_index(dataset: HummanPreprocessedDatasetV2, sequence: str, camera: str) -> Dict[int, int]:
    index_map: Dict[int, int] = {}
    for ds_idx, item in enumerate(dataset.data_list):
        if item["seq_name"] != sequence:
            continue
        sample = dataset[ds_idx]
        frame_id = int(sample["selected_frame_ids"]["rgb"][camera][0])
        index_map[frame_id] = ds_idx
    return index_map


def _select_frame_ids(
    sequence: str,
    camera: str,
    frame_index: Dict[int, int],
    mask_root: Path,
    frame_ids_arg: List[int] | None,
    num_samples: int,
) -> List[int]:
    available = sorted(
        frame_id
        for frame_id in frame_index
        if resolve_humman_mask_path(mask_root, sequence, camera, frame_id).is_file()
    )
    if not available:
        raise RuntimeError(f"No mask-backed frames found for {sequence} {camera}.")
    if frame_ids_arg is not None:
        missing = [frame_id for frame_id in frame_ids_arg if frame_id not in available]
        if missing:
            raise FileNotFoundError(
                f"Requested frames are missing masks or samples for {sequence} {camera}: {missing}"
            )
        return list(frame_ids_arg)
    if num_samples <= 0:
        raise ValueError(f"--num-samples must be > 0, got {num_samples}")
    if num_samples >= len(available):
        return available
    idxs = np.linspace(0, len(available) - 1, num=num_samples, dtype=np.int64)
    return [available[i] for i in idxs.tolist()]


def _export_one_sample(
    out_dir: Path,
    sequence: str,
    camera: str,
    frame_id: int,
    plain_sample: Dict,
    masked_sample: Dict,
    mask_root: Path,
) -> Dict[str, int | str]:
    rgb = plain_sample["input_rgb"][0]
    rgb_masked = masked_sample["input_rgb"][0]
    depth = plain_sample["input_depth"][0]
    depth_masked = masked_sample["input_depth"][0]
    lidar = plain_sample["input_lidar"][0]
    lidar_masked = masked_sample["input_lidar"][0]

    mask_path = resolve_humman_mask_path(mask_root, sequence, camera, frame_id)
    mask_bool = load_humman_binary_mask(mask_path, rgb.shape[:2])
    overlay_bgr = _mask_overlay(rgb, mask_bool)
    depth_bgr = _depth_preview(depth)
    depth_masked_bgr = _depth_preview(depth_masked)

    frame_dir = out_dir / sequence / camera / f"{frame_id:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(frame_dir / "rgb_input.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(frame_dir / "rgb_mask_overlay.jpg"), overlay_bgr)
    cv2.imwrite(str(frame_dir / "rgb_masked.jpg"), cv2.cvtColor(rgb_masked, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(frame_dir / "mask.png"), (mask_bool.astype(np.uint8) * 255))
    cv2.imwrite(str(frame_dir / "depth_input_preview_jet.jpg"), depth_bgr)
    cv2.imwrite(str(frame_dir / "depth_masked_preview_jet.jpg"), depth_masked_bgr)
    np.save(frame_dir / "pointcloud_unmasked.npy", np.asarray(lidar[:, :3], dtype=np.float32))
    np.save(frame_dir / "pointcloud_masked.npy", np.asarray(lidar_masked[:, :3], dtype=np.float32))
    _write_ply_ascii(frame_dir / "pointcloud_unmasked.ply", lidar)
    _write_ply_ascii(frame_dir / "pointcloud_masked.ply", lidar_masked)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB Input")
    axes[0, 1].imshow(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("RGB Mask Overlay")
    axes[0, 2].imshow(rgb_masked)
    axes[0, 2].set_title("RGB Masked")
    axes[0, 3].imshow(mask_bool, cmap="gray")
    axes[0, 3].set_title("Mask")
    axes[1, 0].imshow(cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Depth Preview")
    axes[1, 1].imshow(cv2.cvtColor(depth_masked_bgr, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Depth Masked Preview")
    _plot_pointcloud(axes[1, 2], lidar, "LiDAR Unmasked")
    _plot_pointcloud(axes[1, 3], lidar_masked, "LiDAR Masked")
    for ax in axes[0, :]:
        ax.axis("off")
    axes[1, 0].axis("off")
    axes[1, 1].axis("off")
    fig.suptitle(f"HuMMan Runtime Mask Validation  seq={sequence}  cam={camera}  frame={frame_id:06d}")
    fig.tight_layout()
    fig.savefig(frame_dir / "panel.png", dpi=160)
    plt.close(fig)

    meta = {
        "sequence": sequence,
        "camera": camera,
        "frame_id": frame_id,
        "mask_path": str(mask_path),
        "rgb_foreground_pixels": int((rgb_masked.sum(axis=2) > 0).sum()),
        "mask_foreground_pixels": int(mask_bool.sum()),
        "depth_unmasked_nonzero": int((depth > 0).sum()),
        "depth_masked_nonzero": int((depth_masked > 0).sum()),
        "lidar_unmasked_points": int(lidar.shape[0]),
        "lidar_masked_points": int(lidar_masked.shape[0]),
    }
    with (frame_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def main() -> None:
    args = _parse_args()
    cameras = _normalize_cameras(args.cameras)
    requested_frame_ids = _parse_frame_ids(args.frame_ids)

    summary: List[Dict[str, int | str]] = []
    for camera in cameras:
        plain_ds, masked_ds = _build_datasets(args.data_root, args.mask_root, camera)
        frame_index = _sequence_frame_index(plain_ds, args.sequence, camera)
        selected_frame_ids = _select_frame_ids(
            sequence=args.sequence,
            camera=camera,
            frame_index=frame_index,
            mask_root=args.mask_root,
            frame_ids_arg=requested_frame_ids,
            num_samples=args.num_samples,
        )

        for frame_id in selected_frame_ids:
            ds_idx = frame_index[frame_id]
            plain_sample = plain_ds[ds_idx]
            masked_sample = masked_ds[ds_idx]
            meta = _export_one_sample(
                out_dir=args.out_dir,
                sequence=args.sequence,
                camera=camera,
                frame_id=frame_id,
                plain_sample=plain_sample,
                masked_sample=masked_sample,
                mask_root=args.mask_root,
            )
            summary.append(meta)

    summary_path = args.out_dir / args.sequence / "export_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"exports": summary}, f, indent=2)

    print(f"[humman-mask-validation] wrote: {args.out_dir / args.sequence}")
    print(f"[humman-mask-validation] summary: {summary_path}")


if __name__ == "__main__":
    main()
