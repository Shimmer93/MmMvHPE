#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _list_depth_files(depth_dir: Path, pattern: str) -> list[Path]:
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")
    files = sorted(p for p in depth_dir.glob(pattern) if p.is_file())
    if not files:
        raise ValueError(f"No files matched pattern `{pattern}` under: {depth_dir}")
    return files


def _read_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to read depth image: {path}")
    if depth.ndim != 2:
        raise ValueError(f"Depth image must be single-channel, got shape {depth.shape} at: {path}")
    return depth


def _sample_files(files: list[Path], num_frames: int, seed: int) -> list[Path]:
    if num_frames <= 0:
        raise ValueError(f"`num_frames` must be > 0, got {num_frames}")
    if num_frames > len(files):
        raise ValueError(
            f"`num_frames` ({num_frames}) exceeds available files ({len(files)})."
        )
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(files), size=num_frames, replace=False)
    idx = np.sort(idx)
    return [files[int(i)] for i in idx]


def _compute_pixelwise_max(depth_files: list[Path]) -> np.ndarray:
    base = _read_depth(depth_files[0])
    max_depth = base.copy()
    base_shape = base.shape
    base_dtype = base.dtype

    for p in depth_files[1:]:
        cur = _read_depth(p)
        if cur.shape != base_shape:
            raise ValueError(
                f"Shape mismatch: expected {base_shape}, got {cur.shape} at {p}"
            )
        if cur.dtype != base_dtype:
            raise ValueError(
                f"Dtype mismatch: expected {base_dtype}, got {cur.dtype} at {p}"
            )
        np.maximum(max_depth, cur, out=max_depth)
    return max_depth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample depth frames from a directory, compute pixel-wise maximum "
            "as background depth map, and save it."
        )
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        required=True,
        help="Directory containing depth frames (e.g., /root/autodl-tmp/panoptic/.../depth/kinect_1).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        required=True,
        help="Number of frames to randomly sample (without replacement).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output path for the background depth map (e.g., .../background_max.png).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="Glob pattern for depth files inside depth-dir.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for frame sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    files = _list_depth_files(args.depth_dir, args.pattern)
    sampled = _sample_files(files, args.num_frames, args.seed)
    bg_max = _compute_pixelwise_max(sampled)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(args.output_path), bg_max)
    if not ok:
        raise RuntimeError(f"Failed to write output image: {args.output_path}")

    print(f"Depth files found: {len(files)}")
    print(f"Frames sampled: {len(sampled)}")
    print(f"Output saved: {args.output_path}")
    print(f"Output shape: {tuple(bg_max.shape)}, dtype: {bg_max.dtype}")


if __name__ == "__main__":
    main()
