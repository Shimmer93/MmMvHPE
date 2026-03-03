#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def _read_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to read depth image: {path}")
    if depth.ndim != 2:
        raise ValueError(f"Depth image must be single-channel, got shape {depth.shape} at: {path}")
    return depth


def _list_depth_files(depth_dir: Path, pattern: str) -> list[Path]:
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")
    files = sorted(p for p in depth_dir.glob(pattern) if p.is_file())
    if not files:
        raise ValueError(f"No files matched pattern `{pattern}` under: {depth_dir}")
    return files


def _ensure_odd(value: int, name: str) -> int:
    v = int(value)
    if v <= 0:
        raise ValueError(f"`{name}` must be > 0, got {v}")
    if v % 2 == 0:
        raise ValueError(f"`{name}` must be odd, got {v}")
    return v


def _postprocess_mask(mask: np.ndarray, open_kernel: int, close_kernel: int, min_area: int) -> np.ndarray:
    out = mask.astype(np.uint8)

    if open_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)

    if min_area > 1:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        keep = np.zeros_like(out)
        for label in range(1, n_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area:
                keep[labels == label] = 1
        out = keep
    return out.astype(bool)


def _compute_adaptive_threshold(valid_diff: np.ndarray, min_diff: float, otsu_clip_percentile: float) -> float:
    pos = valid_diff[valid_diff > 0]
    if pos.size < 64:
        return float(min_diff)
    clip_max = float(np.percentile(pos, float(otsu_clip_percentile)))
    if not np.isfinite(clip_max) or clip_max <= 0:
        return float(min_diff)

    # Otsu on clipped/normalized positive diff values to separate noise vs foreground.
    clipped = np.clip(pos, 0.0, clip_max)
    norm = np.round((clipped / clip_max) * 255.0).astype(np.uint8)
    otsu_thr_u8, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_thr = clip_max * (float(otsu_thr_u8) / 255.0)
    return float(max(min_diff, otsu_thr))


def _extract_foreground_mask(
    depth: np.ndarray,
    bg: np.ndarray,
    min_diff: float,
    threshold_mode: str,
    otsu_clip_percentile: float,
    open_kernel: int,
    close_kernel: int,
    min_area: int,
) -> tuple[np.ndarray, float]:
    if depth.shape != bg.shape:
        raise ValueError(f"Shape mismatch: depth={depth.shape}, bg={bg.shape}")
    if depth.dtype != bg.dtype:
        raise ValueError(f"Dtype mismatch: depth={depth.dtype}, bg={bg.dtype}")

    valid_depth = depth > 0
    valid_bg = bg > 0
    valid = valid_depth & valid_bg
    diff = bg.astype(np.float32) - depth.astype(np.float32)

    if threshold_mode == "fixed":
        threshold = float(min_diff)
    elif threshold_mode == "otsu":
        threshold = _compute_adaptive_threshold(
            valid_diff=diff[valid],
            min_diff=float(min_diff),
            otsu_clip_percentile=float(otsu_clip_percentile),
        )
    else:
        raise ValueError(f"Unsupported threshold_mode={threshold_mode}")

    # Foreground is closer than background by at least threshold.
    fg = valid & (diff >= threshold)
    # If background is invalid at a pixel, keep valid depth as foreground.
    fg |= valid_depth & (~valid_bg)

    fg = _postprocess_mask(fg, open_kernel=open_kernel, close_kernel=close_kernel, min_area=min_area)
    return fg, threshold


_WORKER_BG: np.ndarray | None = None
_WORKER_MIN_DIFF: float | None = None
_WORKER_THRESHOLD_MODE: str | None = None
_WORKER_OTSU_CLIP_PERCENTILE: float | None = None
_WORKER_OPEN_KERNEL: int | None = None
_WORKER_CLOSE_KERNEL: int | None = None
_WORKER_MIN_AREA: int | None = None


def _init_worker(
    bg_path: str,
    min_diff: float,
    threshold_mode: str,
    otsu_clip_percentile: float,
    open_kernel: int,
    close_kernel: int,
    min_area: int,
) -> None:
    global _WORKER_BG
    global _WORKER_MIN_DIFF
    global _WORKER_THRESHOLD_MODE
    global _WORKER_OTSU_CLIP_PERCENTILE
    global _WORKER_OPEN_KERNEL
    global _WORKER_CLOSE_KERNEL
    global _WORKER_MIN_AREA

    _WORKER_BG = _read_depth(Path(bg_path))
    _WORKER_MIN_DIFF = float(min_diff)
    _WORKER_THRESHOLD_MODE = str(threshold_mode)
    _WORKER_OTSU_CLIP_PERCENTILE = float(otsu_clip_percentile)
    _WORKER_OPEN_KERNEL = int(open_kernel)
    _WORKER_CLOSE_KERNEL = int(close_kernel)
    _WORKER_MIN_AREA = int(min_area)


def _process_one_depth_map(depth_path_str: str, output_dir_str: str) -> float:
    if _WORKER_BG is None:
        raise RuntimeError("Worker background is not initialized.")
    depth_path = Path(depth_path_str)
    output_dir = Path(output_dir_str)
    depth = _read_depth(depth_path)
    fg_mask, used_thr = _extract_foreground_mask(
        depth=depth,
        bg=_WORKER_BG,
        min_diff=float(_WORKER_MIN_DIFF),
        threshold_mode=str(_WORKER_THRESHOLD_MODE),
        otsu_clip_percentile=float(_WORKER_OTSU_CLIP_PERCENTILE),
        open_kernel=int(_WORKER_OPEN_KERNEL),
        close_kernel=int(_WORKER_CLOSE_KERNEL),
        min_area=int(_WORKER_MIN_AREA),
    )
    subject_only = np.zeros_like(depth)
    subject_only[fg_mask] = depth[fg_mask]
    out_path = output_dir / depth_path.name
    ok = cv2.imwrite(str(out_path), subject_only)
    if not ok:
        raise RuntimeError(f"Failed to write output depth map: {out_path}")
    return float(used_thr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove background from depth maps using a background depth map and save subject-only depth maps."
        )
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        required=True,
        help="Input depth map directory.",
    )
    parser.add_argument(
        "--bg-path",
        type=Path,
        required=True,
        help="Background depth map path (same size/dtype as input depth maps).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for subject-only depth maps.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="Glob pattern for input depth files.",
    )
    parser.add_argument(
        "--min-diff",
        type=float,
        default=30.0,
        help="Lower bound for foreground threshold in raw depth units (uint16 depth usually in millimeters).",
    )
    parser.add_argument(
        "--threshold-mode",
        type=str,
        default="otsu",
        choices=["otsu", "fixed"],
        help="Foreground threshold strategy: per-frame Otsu on (bg-depth) or fixed threshold.",
    )
    parser.add_argument(
        "--otsu-clip-percentile",
        type=float,
        default=99.5,
        help="Clip percentile for positive (bg-depth) values before Otsu (used only in otsu mode).",
    )
    parser.add_argument(
        "--open-kernel",
        type=int,
        default=3,
        help="Odd kernel size for morphological opening (1 disables).",
    )
    parser.add_argument(
        "--close-kernel",
        type=int,
        default=5,
        help="Odd kernel size for morphological closing (1 disables).",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=200,
        help="Minimum connected-component area to keep in foreground mask.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 1))),
        help="Number of worker processes for per-frame extraction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.min_diff < 0:
        raise ValueError(f"`min-diff` must be >= 0, got {args.min_diff}")
    if args.min_area <= 0:
        raise ValueError(f"`min-area` must be > 0, got {args.min_area}")
    if args.workers <= 0:
        raise ValueError(f"`workers` must be > 0, got {args.workers}")
    if not (0.0 < args.otsu_clip_percentile <= 100.0):
        raise ValueError(
            f"`otsu-clip-percentile` must be in (0, 100], got {args.otsu_clip_percentile}"
        )
    open_kernel = _ensure_odd(args.open_kernel, "open-kernel")
    close_kernel = _ensure_odd(args.close_kernel, "close-kernel")

    if not args.bg_path.is_file():
        raise FileNotFoundError(f"Background map not found: {args.bg_path}")

    depth_files = _list_depth_files(args.depth_dir, args.pattern)
    bg = _read_depth(args.bg_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    used_thresholds: list[float] = []
    if args.workers == 1:
        for depth_path in tqdm(depth_files, desc="Extracting foreground depth", unit="frame"):
            depth = _read_depth(depth_path)
            fg_mask, used_thr = _extract_foreground_mask(
                depth=depth,
                bg=bg,
                min_diff=args.min_diff,
                threshold_mode=args.threshold_mode,
                otsu_clip_percentile=args.otsu_clip_percentile,
                open_kernel=open_kernel,
                close_kernel=close_kernel,
                min_area=args.min_area,
            )
            subject_only = np.zeros_like(depth)
            subject_only[fg_mask] = depth[fg_mask]
            out_path = args.output_dir / depth_path.name
            ok = cv2.imwrite(str(out_path), subject_only)
            if not ok:
                raise RuntimeError(f"Failed to write output depth map: {out_path}")
            written += 1
            used_thresholds.append(float(used_thr))
    else:
        with ProcessPoolExecutor(
            max_workers=int(args.workers),
            initializer=_init_worker,
            initargs=(
                str(args.bg_path),
                float(args.min_diff),
                str(args.threshold_mode),
                float(args.otsu_clip_percentile),
                int(open_kernel),
                int(close_kernel),
                int(args.min_area),
            ),
        ) as executor:
            future_to_path = {
                executor.submit(_process_one_depth_map, str(depth_path), str(args.output_dir)): depth_path
                for depth_path in depth_files
            }
            for future in tqdm(
                as_completed(future_to_path),
                total=len(future_to_path),
                desc="Extracting foreground depth",
                unit="frame",
            ):
                depth_path = future_to_path[future]
                try:
                    used_thr = float(future.result())
                except Exception as exc:
                    raise RuntimeError(f"Failed processing depth map: {depth_path}") from exc
                written += 1
                used_thresholds.append(used_thr)

    print(f"Input depth maps: {len(depth_files)}")
    print(f"Background map: {args.bg_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Output files written: {written}")
    if used_thresholds:
        ths = np.asarray(used_thresholds, dtype=np.float32)
        print(
            "Foreground threshold stats: "
            f"min={ths.min():.2f}, median={np.median(ths):.2f}, max={ths.max():.2f} "
            f"(mode={args.threshold_mode}, min_diff={args.min_diff})"
        )


if __name__ == "__main__":
    main()
