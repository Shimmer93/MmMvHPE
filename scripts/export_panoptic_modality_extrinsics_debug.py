#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for p in list(sys.path):
    if p.endswith("/models/pc_encoders/modules"):
        sys.path.remove(p)

from misc.registry import create_dataset


FRAME_ID_RE = re.compile(r"_(\d{8})$")


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _extract_lidar_points(input_lidar: Any) -> np.ndarray:
    arr = _to_numpy(input_lidar)
    if arr.ndim == 4:
        pts = arr[0, -1]
    elif arr.ndim == 3:
        pts = arr[-1]
    elif arr.ndim == 2:
        pts = arr
    else:
        raise ValueError(f"Unsupported input_lidar shape: {arr.shape}")
    if pts.shape[-1] < 3:
        raise ValueError(f"input_lidar last dim must be >=3, got shape {pts.shape}")
    return np.asarray(pts[:, :3], dtype=np.float32)


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


def _camera_to_world(points_cam: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    ext = np.asarray(extrinsic, dtype=np.float32)
    if ext.shape != (3, 4):
        raise ValueError(f"extrinsic must be (3,4), got {ext.shape}")
    rot = ext[:, :3]
    trans = ext[:, 3]
    pts = np.asarray(points_cam, dtype=np.float32)
    return (pts - trans[None, :]) @ rot


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export a Panoptic sample with loaded RGB/depth/LiDAR extrinsics and point clouds "
            "for geometry debugging."
        )
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/exp/panoptic/cross_camera_split/hpe.yml"),
        help="Experiment config containing Panoptic dataset definitions.",
    )
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--index", type=int, default=0, help="Sample index in selected dataset.")
    p.add_argument("--sequence", type=str, default=None, help="Optional single-sequence override.")
    p.add_argument("--camera", type=str, default=None, help="Optional single camera override.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("logs/panoptic_modality_extrinsics_debug"),
        help="Output root.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_key = f"{args.split}_dataset"
    if dataset_key not in cfg:
        raise KeyError(f"Missing `{dataset_key}` in config: {args.config}")
    dataset_cfg = cfg[dataset_key]
    dataset_name = dataset_cfg["name"]
    dataset_params = dict(dataset_cfg.get("params", {}))

    if args.sequence is not None:
        dataset_params["sequence_allowlist"] = [args.sequence]
    if args.camera is not None:
        dataset_params["rgb_cameras"] = [args.camera]
        dataset_params["depth_cameras"] = [args.camera]
        dataset_params["rgb_cameras_per_sample"] = 1
        dataset_params["depth_cameras_per_sample"] = 1
        dataset_params["lidar_cameras_per_sample"] = 1
    # This is a geometry debug utility; use direct sequence/camera selection rather than split-camera filters.
    dataset_params["split_config"] = None
    dataset_params["convert_depth_to_lidar"] = True

    dataset, _ = create_dataset(dataset_name, dataset_params, [])
    if len(dataset) == 0:
        raise ValueError("Selected dataset has zero samples.")
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"--index out of range: {args.index}, dataset size={len(dataset)}")

    sample = dataset[args.index]
    if "input_lidar" not in sample:
        raise KeyError("Sample has no `input_lidar` after forcing convert_depth_to_lidar=True.")

    rgb_cam = sample.get("rgb_camera")
    depth_cam = sample.get("depth_camera")
    lidar_cam = sample.get("lidar_camera")
    if rgb_cam is None or lidar_cam is None:
        raise ValueError("Sample is missing required rgb_camera or lidar_camera metadata.")

    rgb_ext = np.asarray(rgb_cam["extrinsic"], dtype=np.float32)
    depth_ext = None if depth_cam is None else np.asarray(depth_cam["extrinsic"], dtype=np.float32)
    lidar_ext = np.asarray(lidar_cam["extrinsic"], dtype=np.float32)
    effective_depth_ext = lidar_ext if depth_ext is None else depth_ext
    lidar_xyz = _extract_lidar_points(sample["input_lidar"])
    lidar_world = _camera_to_world(lidar_xyz, lidar_ext)
    lidar_world_as_rgb = _camera_to_world(lidar_xyz, rgb_ext)

    seq_name = str(sample.get("seq_name", "unknown_seq"))
    sample_id = str(sample.get("sample_id", f"{args.split}_{args.index}"))
    frame_match = FRAME_ID_RE.search(sample_id)
    frame_id = int(frame_match.group(1)) if frame_match is not None else -1
    selected = sample.get("selected_cameras", {})
    rgb_name = (selected.get("rgb") or [None])[0]
    depth_name = (selected.get("depth") or [None])[0]

    out_dir = args.out_dir / seq_name / f"{frame_id:08d}_{rgb_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_ply_ascii(out_dir / "lidar_sensor_points.ply", lidar_xyz)
    _write_ply_ascii(out_dir / "lidar_world_from_depth_extrinsic.ply", lidar_world)
    _write_ply_ascii(out_dir / "lidar_world_from_rgb_extrinsic_legacy_compare.ply", lidar_world_as_rgb)

    rel_depth_vs_rgb = None
    rel_depth_vs_rgb = (effective_depth_ext - rgb_ext).tolist()

    meta = {
        "config": str(args.config),
        "split": args.split,
        "index": args.index,
        "sample_id": sample_id,
        "sequence": seq_name,
        "frame_id": frame_id,
        "selected_rgb_camera": rgb_name,
        "selected_depth_camera": depth_name,
        "rgb_extrinsic": rgb_ext.tolist(),
        "depth_extrinsic": None if depth_ext is None else depth_ext.tolist(),
        "effective_depth_based_extrinsic": effective_depth_ext.tolist(),
        "lidar_extrinsic": lidar_ext.tolist(),
        "depth_minus_rgb_extrinsic": rel_depth_vs_rgb,
        "num_lidar_points": int(lidar_xyz.shape[0]),
        "sensor_pointcloud_ply": str(out_dir / "lidar_sensor_points.ply"),
        "world_pointcloud_depth_ply": str(out_dir / "lidar_world_from_depth_extrinsic.ply"),
        "world_pointcloud_rgb_legacy_ply": str(out_dir / "lidar_world_from_rgb_extrinsic_legacy_compare.ply"),
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[panoptic-extrinsics-debug] wrote: {out_dir}")
    print(f"[panoptic-extrinsics-debug] meta: {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
