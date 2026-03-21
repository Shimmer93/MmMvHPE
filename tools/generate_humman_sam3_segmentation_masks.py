#!/usr/bin/env python3
"""Generate HuMMan-cropped SAM3 person masks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HUMMAN_RGB_RE = re.compile(
    r"^(?P<sequence>p\d+_a\d+)_kinect_(?P<camera>\d{3})_(?P<frame>\d+)(?P<suffix>\.[A-Za-z0-9]+)$"
)


@dataclass
class PairRunStats:
    sequence: str
    camera: str
    total_images: int = 0
    processed_images: int = 0
    skipped_images: int = 0
    failed_images: int = 0


@dataclass(frozen=True)
class SequenceCameraPair:
    sequence: str
    camera: str


def _fail(message: str) -> None:
    raise SystemExit(f"[humman-sam3-mask] ERROR: {message}")


def _parse_csv_arg(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HuMMan-cropped SAM3 segmentation masks",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="HuMMan-cropped dataset root containing `rgb/`.",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=None,
        help="Optional external output root for masks. Defaults to `<data_root>/sam_segmentation_mask`.",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        default=None,
        help="Optional comma-separated HuMMan sequence ids, e.g. p000441_a000701.",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help="Optional comma-separated camera names, e.g. kinect_007,kinect_008.",
    )
    parser.add_argument(
        "--segmentor-path",
        type=Path,
        default=Path("/opt/data/SAM3_checkpoint"),
        help="Path to `sam3.pt` or a directory containing `sam3.pt`.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device passed to SAM3 loader. Defaults to `cuda`.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mask files instead of skipping them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record per-item failures and continue processing later images.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write the final run summary as JSON.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of disjoint sequence-camera shards. Defaults to 1.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index in [0, num_shards). Defaults to 0.",
    )
    return parser.parse_args()


def _load_human_segmentor(segmentor_path: Path, device: str):
    sam3d_root = REPO_ROOT / "third_party" / "sam-3d-body"
    if not sam3d_root.is_dir():
        _fail(f"SAM-3D-Body submodule not found: {sam3d_root}")
    if str(sam3d_root) not in sys.path:
        sys.path.insert(0, str(sam3d_root))

    build_sam_path = sam3d_root / "tools" / "build_sam.py"
    if not build_sam_path.is_file():
        _fail(f"Missing build_sam.py: {build_sam_path}")

    spec = importlib.util.spec_from_file_location("sam3d_build_sam_humman_masks", build_sam_path)
    if spec is None or spec.loader is None:
        _fail(f"Failed to load SAM3 build module from: {build_sam_path}")
    build_sam_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_sam_module)

    HumanSegmentor = build_sam_module.HumanSegmentor
    try:
        return HumanSegmentor(name="sam3", device=device, path=str(segmentor_path))
    except Exception as exc:
        _fail(f"failed to initialize SAM3 HumanSegmentor: {exc}")


def _validate_startup(args: argparse.Namespace) -> None:
    if not args.data_root.is_dir():
        _fail(f"data root does not exist or is not a directory: {args.data_root}")
    rgb_root = args.data_root / "rgb"
    if not rgb_root.is_dir():
        _fail(f"missing RGB directory: {rgb_root}")
    mask_root = _resolve_mask_root(args.data_root, args.mask_root)
    if not mask_root.parent.exists():
        _fail(f"mask root parent does not exist: {mask_root.parent}")
    if not mask_root.parent.is_dir():
        _fail(f"mask root parent is not a directory: {mask_root.parent}")
    try:
        import torch  # noqa: F401
    except Exception as exc:
        _fail(f"failed to import torch: {exc}")
    if args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            _fail("CUDA device requested but torch.cuda.is_available() is false")
    if args.num_shards < 1:
        _fail(f"--num-shards must be >= 1, got {args.num_shards}")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        _fail(
            f"--shard-index must satisfy 0 <= shard-index < num-shards, got "
            f"{args.shard_index} with num_shards={args.num_shards}"
        )
    segmentor_path = args.segmentor_path.expanduser()
    checkpoint_path = segmentor_path / "sam3.pt" if segmentor_path.is_dir() else segmentor_path
    if not checkpoint_path.exists():
        _fail(
            f"SAM3 checkpoint not found at {checkpoint_path}. "
            "Pass `--segmentor-path` as a `.pt` file or a directory containing `sam3.pt`."
        )


def _normalize_camera_name(raw: str) -> str:
    token = raw.strip()
    if token.startswith("kinect_"):
        suffix = token.split("_")[-1]
    else:
        suffix = token
    return f"kinect_{int(suffix):03d}"


def _parse_humman_rgb_name(image_path: Path) -> tuple[str, str, int]:
    match = HUMMAN_RGB_RE.match(image_path.name)
    if match is None:
        raise ValueError(f"unexpected HuMMan RGB filename: {image_path.name}")
    return (
        match.group("sequence"),
        f"kinect_{match.group('camera')}",
        int(match.group("frame")),
    )


def _list_rgb_images(rgb_root: Path) -> list[Path]:
    return sorted(
        path
        for path in rgb_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _build_pair_to_images(
    data_root: Path,
    requested_sequences: list[str] | None,
    requested_cameras: list[str] | None,
) -> dict[SequenceCameraPair, list[Path]]:
    rgb_root = data_root / "rgb"
    if not rgb_root.is_dir():
        raise FileNotFoundError(f"missing RGB directory: {rgb_root}")

    allowed_sequences = set(requested_sequences) if requested_sequences is not None else None
    allowed_cameras = (
        {_normalize_camera_name(name) for name in requested_cameras}
        if requested_cameras is not None
        else None
    )
    grouped: dict[SequenceCameraPair, list[Path]] = {}
    for image_path in _list_rgb_images(rgb_root):
        sequence, camera, _frame = _parse_humman_rgb_name(image_path)
        if allowed_sequences is not None and sequence not in allowed_sequences:
            continue
        if allowed_cameras is not None and camera not in allowed_cameras:
            continue
        pair = SequenceCameraPair(sequence=sequence, camera=camera)
        grouped.setdefault(pair, []).append(image_path)
    return {pair: paths for pair, paths in sorted(grouped.items(), key=lambda item: (item[0].sequence, item[0].camera))}


def _select_shard_pairs(
    pairs: list[SequenceCameraPair],
    *,
    num_shards: int,
    shard_index: int,
) -> list[SequenceCameraPair]:
    return [pair for idx, pair in enumerate(pairs) if idx % num_shards == shard_index]


def _mask_output_name(image_path: Path) -> str:
    return f"{image_path.stem}.png"


def _resolve_mask_path(data_root: Path, image_path: Path) -> Path:
    return data_root / "sam_segmentation_mask" / _mask_output_name(image_path)


def _resolve_mask_root(data_root: Path, mask_root: Path | None) -> Path:
    if mask_root is None:
        return data_root / "sam_segmentation_mask"
    return mask_root.expanduser()


def _resolve_mask_path_from_root(mask_root: Path, image_path: Path) -> Path:
    return mask_root / _mask_output_name(image_path)


def _read_bgr_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode image: {image_path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected BGR image with shape (H,W,3), got {image.shape} for {image_path}")
    return image


def _union_masks(masks: np.ndarray | list[np.ndarray], height: int, width: int) -> np.ndarray:
    if masks is None:
        return np.zeros((height, width), dtype=np.uint8)
    arr = np.asarray(masks)
    if arr.size == 0:
        return np.zeros((height, width), dtype=np.uint8)
    if arr.ndim == 2:
        union = arr.astype(bool)
    elif arr.ndim == 3:
        union = np.any(arr.astype(bool), axis=0)
    else:
        raise ValueError(f"unexpected SAM3 mask shape: {arr.shape}")
    return union.astype(np.uint8) * 255


def _write_binary_mask(mask_path: Path, mask: np.ndarray) -> None:
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if mask.dtype != np.uint8:
        raise ValueError(f"mask must be uint8 before write, got {mask.dtype}")
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape (H,W), got {mask.shape}")
    ok = cv2.imwrite(str(mask_path), mask)
    if not ok:
        raise IOError(f"cv2.imwrite failed for {mask_path}")


def _record_failure(
    failures: list[dict],
    *,
    sequence: str,
    camera: str,
    image_path: Path | None,
    reason: str,
) -> None:
    failures.append(
        {
            "sequence": sequence,
            "camera": camera,
            "image_path": str(image_path) if image_path is not None else None,
            "reason": reason,
        }
    )


def _process_image(
    segmentor,
    image_path: Path,
    mask_path: Path,
    *,
    overwrite: bool,
) -> tuple[str, str | None]:
    if mask_path.exists() and not overwrite:
        return "skipped", None
    image = _read_bgr_image(image_path)
    masks, _scores = segmentor.run_sam(image, np.zeros((1, 4), dtype=np.float32))
    merged_mask = _union_masks(masks, image.shape[0], image.shape[1])
    _write_binary_mask(mask_path, merged_mask)
    return "processed", None


def _handle_pair(
    segmentor,
    mask_root: Path,
    pair: SequenceCameraPair,
    image_paths: list[Path],
    *,
    overwrite: bool,
    continue_on_error: bool,
    failures: list[dict],
    overall_progress: tqdm | None = None,
) -> PairRunStats:
    if not image_paths:
        raise FileNotFoundError(f"no RGB images found for {pair.sequence}:{pair.camera}")

    stats = PairRunStats(sequence=pair.sequence, camera=pair.camera, total_images=len(image_paths))
    progress = tqdm(image_paths, desc=f"{pair.sequence}:{pair.camera}", unit="img", dynamic_ncols=True)
    for image_path in progress:
        mask_path = _resolve_mask_path_from_root(mask_root, image_path)
        try:
            status, _ = _process_image(
                segmentor,
                image_path,
                mask_path,
                overwrite=overwrite,
            )
            if status == "skipped":
                stats.skipped_images += 1
            else:
                stats.processed_images += 1
            if overall_progress is not None:
                overall_progress.update(1)
                overall_progress.set_postfix(
                    pair=f"{pair.sequence}:{pair.camera}",
                    failed=len(failures),
                )
            progress.set_postfix(
                processed=stats.processed_images,
                skipped=stats.skipped_images,
                failed=stats.failed_images,
            )
        except Exception as exc:
            stats.failed_images += 1
            if overall_progress is not None:
                overall_progress.update(1)
            _record_failure(
                failures,
                sequence=pair.sequence,
                camera=pair.camera,
                image_path=image_path,
                reason=str(exc),
            )
            if overall_progress is not None:
                overall_progress.set_postfix(
                    pair=f"{pair.sequence}:{pair.camera}",
                    failed=len(failures),
                )
            progress.set_postfix(
                processed=stats.processed_images,
                skipped=stats.skipped_images,
                failed=stats.failed_images,
            )
            if not continue_on_error:
                raise
    return stats


def _build_summary(
    *,
    data_root: Path,
    mask_root: Path,
    requested_sequences: list[str] | None,
    requested_cameras: list[str] | None,
    num_shards: int,
    shard_index: int,
    assigned_pairs: list[SequenceCameraPair],
    overwrite: bool,
    continue_on_error: bool,
    pair_stats: list[PairRunStats],
    failures: list[dict],
) -> dict:
    total_images = sum(item.total_images for item in pair_stats)
    processed_images = sum(item.processed_images for item in pair_stats)
    skipped_images = sum(item.skipped_images for item in pair_stats)
    failed_images = sum(item.failed_images for item in pair_stats)
    return {
        "data_root": str(data_root),
        "mask_root": str(mask_root),
        "requested_sequences": requested_sequences,
        "requested_cameras": requested_cameras,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "assigned_sequence_camera_pairs": [
            {"sequence": pair.sequence, "camera": pair.camera} for pair in assigned_pairs
        ],
        "overwrite": overwrite,
        "continue_on_error": continue_on_error,
        "pair_runs": [
            {
                "sequence": item.sequence,
                "camera": item.camera,
                "total_images": item.total_images,
                "processed_images": item.processed_images,
                "skipped_images": item.skipped_images,
                "failed_images": item.failed_images,
            }
            for item in pair_stats
        ],
        "totals": {
            "total_images": total_images,
            "processed_images": processed_images,
            "skipped_images": skipped_images,
            "failed_images": failed_images,
        },
        "failures": failures,
    }


def _print_summary(summary: dict) -> None:
    totals = summary["totals"]
    print(
        "[humman-sam3-mask] completed "
        f"shard={summary['shard_index']}/{summary['num_shards']} "
        f"total_images={totals['total_images']} "
        f"processed={totals['processed_images']} "
        f"skipped={totals['skipped_images']} "
        f"failed={totals['failed_images']}"
    )
    if summary["failures"]:
        print("[humman-sam3-mask] failures:")
        for item in summary["failures"]:
            print(
                "[humman-sam3-mask] "
                f"{item['sequence']}:{item['camera']}:{item['image_path']}: {item['reason']}"
            )


def main() -> None:
    args = parse_args()
    requested_sequences = _parse_csv_arg(args.sequences)
    requested_cameras = _parse_csv_arg(args.cameras)
    _validate_startup(args)
    mask_root = _resolve_mask_root(args.data_root, args.mask_root)

    pair_to_images = _build_pair_to_images(args.data_root, requested_sequences, requested_cameras)
    if not pair_to_images:
        _fail("no matching HuMMan RGB images found for the requested filters")
    all_pairs = list(pair_to_images.keys())
    assigned_pairs = _select_shard_pairs(
        all_pairs,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    if not assigned_pairs:
        _fail(
            f"no sequence-camera pairs assigned to shard {args.shard_index} "
            f"out of {args.num_shards} shard(s)"
        )
    print(
        f"[humman-sam3-mask] shard {args.shard_index}/{args.num_shards}: "
        f"{len(assigned_pairs)} sequence-camera pair(s) assigned"
    )
    assigned_total_images = sum(len(pair_to_images[pair]) for pair in assigned_pairs)
    print(
        f"[humman-sam3-mask] shard {args.shard_index}/{args.num_shards}: "
        f"{assigned_total_images} image(s) assigned"
    )

    segmentor = _load_human_segmentor(args.segmentor_path.expanduser(), args.device)
    pair_stats: list[PairRunStats] = []
    failures: list[dict] = []
    overall_progress = tqdm(
        total=assigned_total_images,
        desc=f"shard {args.shard_index}/{args.num_shards} overall",
        unit="img",
        dynamic_ncols=True,
    )
    try:
        for pair in assigned_pairs:
            try:
                stats = _handle_pair(
                    segmentor,
                    mask_root,
                    pair,
                    pair_to_images[pair],
                    overwrite=args.overwrite,
                    continue_on_error=args.continue_on_error,
                    failures=failures,
                    overall_progress=overall_progress,
                )
                pair_stats.append(stats)
            except Exception as exc:
                _record_failure(
                    failures,
                    sequence=pair.sequence,
                    camera=pair.camera,
                    image_path=None,
                    reason=str(exc),
                )
                if not args.continue_on_error:
                    raise
    finally:
        overall_progress.close()

    summary = _build_summary(
        data_root=args.data_root,
        mask_root=mask_root,
        requested_sequences=requested_sequences,
        requested_cameras=requested_cameras,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        assigned_pairs=assigned_pairs,
        overwrite=args.overwrite,
        continue_on_error=args.continue_on_error,
        pair_stats=pair_stats,
        failures=failures,
    )
    _print_summary(summary)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
