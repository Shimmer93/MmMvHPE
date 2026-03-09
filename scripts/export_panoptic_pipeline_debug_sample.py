#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

# Ensure this repo root takes precedence on sys.path so `misc.*` and `datasets.*`
# resolve to project modules even when external paths inject conflicting modules.
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
    # Expected shapes:
    # - (T, N, C)
    # - (V, T, N, C)
    # - (N, C)
    if arr.ndim == 4:
        pts = arr[0, -1]  # first view, last frame
    elif arr.ndim == 3:
        pts = arr[-1]  # last frame
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export one Panoptic sample from the configured data pipeline: "
            "pipeline point cloud + matching raw depth/rgb files."
        )
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/exp/panoptic/cross_camera_split/hpe.yml"),
        help="Experiment config containing dataset and pipeline definitions.",
    )
    p.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Which dataset block/pipeline to use from config.",
    )
    p.add_argument("--index", type=int, default=0, help="Sample index in selected dataset.")
    p.add_argument("--out-dir", type=Path, default=Path("logs/panoptic_pipeline_debug"))
    p.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Optional single-sequence override (sets sequence_allowlist=[sequence]).",
    )
    p.add_argument(
        "--no-convert-depth-to-lidar",
        action="store_true",
        help="Disable convert_depth_to_lidar override if config has it enabled.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_key = f"{args.split}_dataset"
    pipeline_key = f"{args.split}_pipeline"
    if dataset_key not in cfg:
        raise KeyError(f"Missing `{dataset_key}` in config: {args.config}")
    if pipeline_key not in cfg:
        raise KeyError(f"Missing `{pipeline_key}` in config: {args.config}")

    dataset_cfg = cfg[dataset_key]
    dataset_name = dataset_cfg["name"]
    dataset_params = dict(dataset_cfg.get("params", {}))
    pipeline_cfg = cfg[pipeline_key]

    if args.sequence is not None:
        dataset_params["sequence_allowlist"] = [args.sequence]
    if args.no_convert_depth_to_lidar:
        dataset_params["convert_depth_to_lidar"] = False

    dataset, _ = create_dataset(dataset_name, dataset_params, pipeline_cfg)
    if len(dataset) == 0:
        raise ValueError("Selected dataset has zero samples.")
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"--index out of range: {args.index}, dataset size={len(dataset)}")

    sample = dataset[args.index]
    if "input_lidar" not in sample:
        raise KeyError(
            "Sample has no `input_lidar`. Ensure modality/pipeline enables depth->lidar conversion."
        )

    sample_id = str(sample.get("sample_id", f"{args.split}_{args.index}"))
    seq_name = str(sample.get("seq_name", "unknown_seq"))
    selected = sample.get("selected_cameras", {})
    rgb_cam = (selected.get("rgb") or [None])[0]
    depth_cam = (selected.get("depth") or [None])[0]
    if rgb_cam is None or depth_cam is None:
        raise ValueError(f"Sample does not contain selected rgb/depth cameras: {selected}")

    frame_match = FRAME_ID_RE.search(sample_id)
    if frame_match is None:
        raise ValueError(f"Could not parse frame id from sample_id: {sample_id}")
    frame_id = int(frame_match.group(1))

    seq_info = dataset.sequence_data[seq_name]
    rgb_path = Path(seq_info["rgb_by_cam"][rgb_cam][frame_id])
    depth_path = Path(seq_info["depth_by_cam"][depth_cam][frame_id])
    if not rgb_path.is_file():
        raise FileNotFoundError(f"RGB file missing: {rgb_path}")
    if not depth_path.is_file():
        raise FileNotFoundError(f"Depth file missing: {depth_path}")

    lidar_xyz = _extract_lidar_points(sample["input_lidar"])

    out_dir = args.out_dir / seq_name / f"{frame_id:08d}_{rgb_cam}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_out = out_dir / "rgb.jpg"
    depth_out = out_dir / "depth.png"
    pc_npy_out = out_dir / "pointcloud_pipeline.npy"
    pc_ply_out = out_dir / "pointcloud_pipeline.ply"
    depth_preview_out = out_dir / "depth_preview_jet.png"
    meta_out = out_dir / "meta.json"

    shutil.copy2(rgb_path, rgb_out)
    shutil.copy2(depth_path, depth_out)
    np.save(pc_npy_out, lidar_xyz)
    _write_ply_ascii(pc_ply_out, lidar_xyz)

    depth_img = cv2.imread(str(depth_out), cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        raise RuntimeError(f"Failed to read depth image: {depth_out}")
    depth_f = depth_img.astype(np.float32)
    nz = depth_f[depth_f > 0]
    if nz.size > 0:
        lo, hi = np.percentile(nz, [2.0, 98.0]).astype(np.float32)
        depth_norm = np.clip((depth_f - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    else:
        lo, hi = 0.0, 1.0
        depth_norm = np.zeros_like(depth_f, dtype=np.float32)
    depth_u8 = (depth_norm * 255.0).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
    cv2.imwrite(str(depth_preview_out), depth_color)

    meta = {
        "config": str(args.config),
        "split": args.split,
        "index": args.index,
        "sample_id": sample_id,
        "sequence": seq_name,
        "frame_id": frame_id,
        "rgb_camera": rgb_cam,
        "depth_camera": depth_cam,
        "source_rgb_path": str(rgb_path),
        "source_depth_path": str(depth_path),
        "pointcloud_num_points": int(lidar_xyz.shape[0]),
        "pointcloud_xyz_min": lidar_xyz.min(axis=0).tolist() if lidar_xyz.size > 0 else [0.0, 0.0, 0.0],
        "pointcloud_xyz_max": lidar_xyz.max(axis=0).tolist() if lidar_xyz.size > 0 else [0.0, 0.0, 0.0],
        "depth_nonzero_count": int(nz.size),
        "depth_preview_percentiles_2_98": [float(lo), float(hi)],
    }
    with meta_out.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[export-panoptic-debug] wrote: {out_dir}")
    print(f"[export-panoptic-debug] rgb: {rgb_out}")
    print(f"[export-panoptic-debug] depth: {depth_out}")
    print(f"[export-panoptic-debug] depth_preview: {depth_preview_out}")
    print(f"[export-panoptic-debug] pointcloud: {pc_ply_out}")
    print(f"[export-panoptic-debug] meta: {meta_out}")


if __name__ == "__main__":
    main()
