#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.transforms.panoptic_mask_transforms import (
    apply_binary_mask_to_frame,
    load_panoptic_binary_mask,
    load_panoptic_camera_meta,
    panoptic_disk_camera_name,
    reproject_rgb_mask_to_depth_mask,
    resolve_panoptic_mask_path,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export Panoptic RGB/depth/mask validation samples with RGB overlays and "
            "masked/unmasked depth point clouds for manual inspection."
        )
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/opt/data/panoptic_kinoptic_single_actor_cropped"),
        help="Root of the preprocessed Panoptic dataset.",
    )
    p.add_argument("--sequence", type=str, required=True, help="Sequence name, e.g. 170915_office1.")
    p.add_argument("--camera", type=str, required=True, help="Camera name, e.g. kinect_8 or kinect_008.")
    p.add_argument(
        "--frame-ids",
        type=str,
        default=None,
        help="Comma-separated 8-digit frame ids. If omitted, frames are selected evenly from available common frames.",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of frames to export when --frame-ids is omitted.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("logs/panoptic_mask_validation"),
        help="Output root for exported validation artifacts.",
    )
    p.add_argument(
        "--depth-subdir",
        type=str,
        default="depth",
        help="Depth source subdirectory under the sequence root, e.g. `depth` or `depth_fg`.",
    )
    return p.parse_args()


def _write_ply_ascii(path: Path, points_xyz: np.ndarray) -> None:
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"points_xyz must be (N,3), got {points_xyz.shape}")
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points_xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points_xyz:
            f.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f}\n")


def _load_camera_meta(sequence_root: Path, camera_name: str) -> Dict[str, np.ndarray]:
    return load_panoptic_camera_meta(sequence_root, camera_name)


def _select_frame_ids(
    sequence_root: Path,
    camera_name: str,
    frame_ids_arg: str | None,
    num_samples: int,
    depth_subdir: str,
) -> List[int]:
    disk_camera = panoptic_disk_camera_name(camera_name)
    rgb_dir = sequence_root / "rgb" / disk_camera
    depth_dir = sequence_root / depth_subdir / disk_camera
    mask_dir = sequence_root / "sam_segmentation_mask" / disk_camera
    for req in (rgb_dir, depth_dir, mask_dir):
        if not req.is_dir():
            raise FileNotFoundError(f"Required directory not found: {req}")

    if frame_ids_arg:
        out = [int(x.strip()) for x in frame_ids_arg.split(",") if x.strip()]
        if not out:
            raise ValueError("--frame-ids was provided but no valid frame ids were parsed.")
        return out

    rgb_stems = {p.stem for p in rgb_dir.glob("*") if p.is_file()}
    depth_stems = {p.stem for p in depth_dir.glob("*.png") if p.is_file()}
    mask_stems = {p.stem for p in mask_dir.glob("*.png") if p.is_file()}
    common = sorted(rgb_stems & depth_stems & mask_stems)
    if not common:
        raise ValueError(f"No common RGB/depth/mask frames under {sequence_root} camera={disk_camera}")
    if num_samples <= 0:
        raise ValueError(f"--num-samples must be > 0, got {num_samples}")
    if num_samples >= len(common):
        return [int(x) for x in common]
    idxs = np.linspace(0, len(common) - 1, num=num_samples, dtype=np.int64)
    return [int(common[i]) for i in idxs.tolist()]


def _depth_preview(depth: np.ndarray) -> np.ndarray:
    depth_f = depth.astype(np.float32)
    nz = depth_f[depth_f > 0]
    if nz.size == 0:
        norm = np.zeros_like(depth_f, dtype=np.float32)
    else:
        lo, hi = np.percentile(nz, [2.0, 98.0]).astype(np.float32)
        norm = np.clip((depth_f - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    depth_u8 = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)


def _mask_overlay(rgb_bgr: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    overlay = rgb_bgr.copy()
    green = np.zeros_like(overlay)
    green[..., 1] = 255
    overlay[mask_bool] = cv2.addWeighted(overlay[mask_bool], 0.5, green[mask_bool], 0.5, 0.0)

    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, contourIdx=-1, color=(0, 0, 255), thickness=1)
    return overlay


def _depth_to_pointcloud(depth: np.ndarray, k_depth: np.ndarray) -> np.ndarray:
    if depth.ndim != 2:
        raise ValueError(f"Depth image must be single-channel, got {depth.shape}")
    ys, xs = np.nonzero(depth > 0)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth[ys, xs].astype(np.float32) / 1000.0
    fx = float(k_depth[0, 0])
    fy = float(k_depth[1, 1])
    cx = float(k_depth[0, 2])
    cy = float(k_depth[1, 2])
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    return np.stack([x, y, z], axis=1).astype(np.float32)


def main() -> None:
    args = _parse_args()
    sequence_root = (args.data_root / args.sequence).expanduser().resolve()
    if not sequence_root.is_dir():
        raise FileNotFoundError(f"Sequence root not found: {sequence_root}")

    camera_name = panoptic_disk_camera_name(args.camera)
    depth_subdir = str(args.depth_subdir).strip().strip("/")
    if not depth_subdir:
        raise ValueError("--depth-subdir must be a non-empty directory name.")
    frame_ids = _select_frame_ids(sequence_root, camera_name, args.frame_ids, args.num_samples, depth_subdir)
    camera_meta = _load_camera_meta(sequence_root, camera_name)

    out_root = args.out_dir / args.sequence / camera_name / depth_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    exported = []
    for frame_id in frame_ids:
        rgb_path = sequence_root / "rgb" / camera_name / f"{frame_id:08d}.jpg"
        depth_path = sequence_root / depth_subdir / camera_name / f"{frame_id:08d}.png"
        mask_path = resolve_panoptic_mask_path(sequence_root, camera_name, frame_id)

        if not rgb_path.is_file():
            raise FileNotFoundError(f"Missing RGB frame: {rgb_path}")
        if not depth_path.is_file():
            raise FileNotFoundError(f"Missing depth frame: {depth_path}")

        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None:
            raise RuntimeError(f"Failed to read RGB frame: {rgb_path}")
        if depth is None:
            raise RuntimeError(f"Failed to read depth frame: {depth_path}")
        if depth.ndim != 2:
            raise ValueError(f"Expected single-channel depth, got shape={depth.shape} at {depth_path}")

        mask_bool = load_panoptic_binary_mask(mask_path, rgb.shape[:2])
        if depth.shape[:2] != rgb.shape[:2]:
            raise ValueError(
                f"RGB/depth shape mismatch for validation export: rgb={rgb.shape}, depth={depth.shape}, frame={frame_id}"
            )

        rgb_masked = apply_binary_mask_to_frame(rgb, mask_bool)
        depth_masked_naive = apply_binary_mask_to_frame(depth, mask_bool)
        color_ext = np.asarray(camera_meta["extrinsic_world_to_color"], dtype=np.float32).copy()
        unit = str(camera_meta["extrinsic_world_to_color_unit"]).lower()
        if unit == "cm":
            color_ext[:, 3] *= 0.01
        elif unit != "m":
            raise ValueError(f"Unsupported extrinsic_world_to_color_unit in validation export: {unit}")
        depth_from_color = np.linalg.inv(np.asarray(camera_meta["M_depth"], dtype=np.float32)) @ np.asarray(
            camera_meta["M_color"], dtype=np.float32
        )
        depth_ext = (depth_from_color @ np.vstack([color_ext, np.array([0, 0, 0, 1], dtype=np.float32)]))[:3, :]
        depth_mask_reprojected = reproject_rgb_mask_to_depth_mask(
            depth_frame=depth.astype(np.float32) / 1000.0,
            rgb_mask=mask_bool,
            k_depth=np.asarray(camera_meta["K_depth"], dtype=np.float32),
            k_color=np.asarray(camera_meta["K_color"], dtype=np.float32),
            depth_extrinsic=depth_ext.astype(np.float32),
            color_extrinsic=color_ext.astype(np.float32),
        )
        depth_masked = apply_binary_mask_to_frame(depth, depth_mask_reprojected)
        rgb_overlay = _mask_overlay(rgb, mask_bool)

        pc_unmasked = _depth_to_pointcloud(depth, camera_meta["K_depth"])
        pc_masked_naive = _depth_to_pointcloud(depth_masked_naive, camera_meta["K_depth"])
        pc_masked = _depth_to_pointcloud(depth_masked, camera_meta["K_depth"])

        frame_dir = out_root / f"{frame_id:08d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(frame_dir / "rgb_input.jpg"), rgb)
        cv2.imwrite(str(frame_dir / "rgb_mask_overlay.jpg"), rgb_overlay)
        cv2.imwrite(str(frame_dir / "rgb_masked.jpg"), rgb_masked)
        cv2.imwrite(str(frame_dir / "mask.png"), (mask_bool.astype(np.uint8) * 255))
        cv2.imwrite(str(frame_dir / "depth_mask_reprojected.png"), (depth_mask_reprojected.astype(np.uint8) * 255))
        cv2.imwrite(str(frame_dir / "depth_input.png"), depth)
        cv2.imwrite(str(frame_dir / "depth_masked_naive.png"), depth_masked_naive)
        cv2.imwrite(str(frame_dir / "depth_masked.png"), depth_masked)
        cv2.imwrite(str(frame_dir / "depth_input_preview_jet.jpg"), _depth_preview(depth))
        cv2.imwrite(str(frame_dir / "depth_masked_naive_preview_jet.jpg"), _depth_preview(depth_masked_naive))
        cv2.imwrite(str(frame_dir / "depth_masked_preview_jet.jpg"), _depth_preview(depth_masked))
        np.save(frame_dir / "pointcloud_unmasked.npy", pc_unmasked)
        np.save(frame_dir / "pointcloud_masked_naive.npy", pc_masked_naive)
        np.save(frame_dir / "pointcloud_masked.npy", pc_masked)
        _write_ply_ascii(frame_dir / "pointcloud_unmasked.ply", pc_unmasked)
        _write_ply_ascii(frame_dir / "pointcloud_masked_naive.ply", pc_masked_naive)
        _write_ply_ascii(frame_dir / "pointcloud_masked.ply", pc_masked)

        meta = {
            "sequence": args.sequence,
            "camera": camera_name,
            "frame_id": frame_id,
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "depth_subdir": depth_subdir,
            "mask_path": str(mask_path),
            "mask_foreground_pixels": int(mask_bool.sum()),
            "depth_mask_reprojected_foreground_pixels": int(depth_mask_reprojected.sum()),
            "pointcloud_unmasked_points": int(pc_unmasked.shape[0]),
            "pointcloud_masked_naive_points": int(pc_masked_naive.shape[0]),
            "pointcloud_masked_points": int(pc_masked.shape[0]),
        }
        with (frame_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        exported.append(meta)

    summary_path = out_root / "export_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"exports": exported}, f, indent=2)

    print(f"[panoptic-mask-validation] wrote: {out_root}")
    print(f"[panoptic-mask-validation] summary: {summary_path}")


if __name__ == "__main__":
    main()
