#!/usr/bin/env python3
"""Generate sequence-local Panoptic SAM3 person masks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class CameraRunStats:
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
    raise SystemExit(f"[panoptic-sam3-mask] ERROR: {message}")


def _parse_csv_arg(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Panoptic SAM3 segmentation masks",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Panoptic dataset root containing per-sequence folders.",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        default=None,
        help="Optional comma-separated sequence names. Defaults to all sequences under data root.",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help="Optional comma-separated camera names, e.g. kinect_1,kinect_4.",
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

    spec = importlib.util.spec_from_file_location("sam3d_build_sam_masks", build_sam_path)
    if spec is None or spec.loader is None:
        _fail(f"Failed to load SAM3 build module from: {build_sam_path}")
    build_sam_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_sam_module)

    HumanSegmentor = build_sam_module.HumanSegmentor
    try:
        return HumanSegmentor(name="sam3", device=device, path=str(segmentor_path))
    except Exception as exc:
        _fail(f"failed to initialize SAM3 HumanSegmentor: {exc}")


def _validate_startup(args: argparse.Namespace):
    if not args.data_root.is_dir():
        _fail(f"data root does not exist or is not a directory: {args.data_root}")
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
    if segmentor_path.is_dir():
        checkpoint_path = segmentor_path / "sam3.pt"
    else:
        checkpoint_path = segmentor_path
    if not checkpoint_path.exists():
        _fail(
            f"SAM3 checkpoint not found at {checkpoint_path}. "
            "Pass `--segmentor-path` as a `.pt` file or a directory containing `sam3.pt`."
        )
    return checkpoint_path


def _resolve_sequence_root(data_root: Path, sequence_name: str) -> Path:
    return data_root / sequence_name


def _resolve_rgb_root(sequence_root: Path) -> Path:
    return sequence_root / "rgb"


def _resolve_camera_rgb_dir(sequence_root: Path, camera_name: str) -> Path:
    return _resolve_rgb_root(sequence_root) / camera_name


def _resolve_camera_mask_dir(sequence_root: Path, camera_name: str) -> Path:
    return sequence_root / "sam_segmentation_mask" / camera_name


def _list_sequences(data_root: Path, requested_sequences: list[str] | None) -> list[str]:
    if requested_sequences is not None:
        return requested_sequences
    return sorted(
        path.name
        for path in data_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def _list_cameras(rgb_root: Path, requested_cameras: list[str] | None) -> list[str]:
    if requested_cameras is not None:
        return requested_cameras
    return sorted(
        path.name
        for path in rgb_root.iterdir()
        if path.is_dir() and path.name.startswith("kinect_")
    )


def _list_rgb_images(camera_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in camera_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _build_sequence_camera_pairs(
    data_root: Path,
    requested_sequences: list[str] | None,
    requested_cameras: list[str] | None,
) -> list[SequenceCameraPair]:
    sequence_names = _list_sequences(data_root, requested_sequences)
    if not sequence_names:
        _fail("no sequences selected for processing")

    pairs: list[SequenceCameraPair] = []
    for sequence_name in sequence_names:
        sequence_root = _resolve_sequence_root(data_root, sequence_name)
        if not sequence_root.is_dir():
            raise FileNotFoundError(f"missing sequence directory: {sequence_root}")
        rgb_root = _resolve_rgb_root(sequence_root)
        if not rgb_root.is_dir():
            raise FileNotFoundError(f"missing RGB root: {rgb_root}")
        camera_names = _list_cameras(rgb_root, requested_cameras)
        if not camera_names:
            raise FileNotFoundError(f"no cameras selected under {rgb_root}")
        for camera_name in camera_names:
            pairs.append(SequenceCameraPair(sequence=sequence_name, camera=camera_name))
    return pairs


def _select_shard_pairs(
    pairs: list[SequenceCameraPair],
    *,
    num_shards: int,
    shard_index: int,
) -> list[SequenceCameraPair]:
    return [pair for idx, pair in enumerate(pairs) if idx % num_shards == shard_index]


def _mask_output_name(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        return image_path.name
    return f"{image_path.stem}.png"


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
    return (union.astype(np.uint8) * 255)


def _write_binary_mask(mask_path: Path, mask: np.ndarray) -> None:
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if mask.dtype != np.uint8:
        raise ValueError(f"mask must be uint8 before write, got {mask.dtype}")
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape (H,W), got {mask.shape}")
    ok = cv2.imwrite(str(mask_path), mask)
    if not ok:
        raise IOError(f"cv2.imwrite failed for {mask_path}")


def _record_failure(failures: list[dict], *, sequence: str, camera: str, image_path: Path | None, reason: str) -> None:
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


def _handle_sequence_camera(
    segmentor,
    sequence_root: Path,
    sequence_name: str,
    camera_name: str,
    *,
    overwrite: bool,
    continue_on_error: bool,
    failures: list[dict],
) -> CameraRunStats:
    camera_dir = _resolve_camera_rgb_dir(sequence_root, camera_name)
    if not camera_dir.is_dir():
        raise FileNotFoundError(f"missing RGB camera directory: {camera_dir}")

    image_paths = _list_rgb_images(camera_dir)
    if not image_paths:
        raise FileNotFoundError(f"no RGB images found under {camera_dir}")

    mask_dir = _resolve_camera_mask_dir(sequence_root, camera_name)
    stats = CameraRunStats(sequence=sequence_name, camera=camera_name, total_images=len(image_paths))
    progress = tqdm(
        image_paths,
        desc=f"{sequence_name}:{camera_name}",
        unit="img",
        dynamic_ncols=True,
    )
    for image_path in progress:
        mask_path = mask_dir / _mask_output_name(image_path)
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
            progress.set_postfix(
                processed=stats.processed_images,
                skipped=stats.skipped_images,
                failed=stats.failed_images,
            )
        except Exception as exc:
            stats.failed_images += 1
            _record_failure(
                failures,
                sequence=sequence_name,
                camera=camera_name,
                image_path=image_path,
                reason=str(exc),
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
    requested_sequences: list[str] | None,
    requested_cameras: list[str] | None,
    num_shards: int,
    shard_index: int,
    assigned_pairs: list[SequenceCameraPair],
    overwrite: bool,
    continue_on_error: bool,
    camera_stats: list[CameraRunStats],
    failures: list[dict],
) -> dict:
    total_images = sum(item.total_images for item in camera_stats)
    processed_images = sum(item.processed_images for item in camera_stats)
    skipped_images = sum(item.skipped_images for item in camera_stats)
    failed_images = sum(item.failed_images for item in camera_stats)
    return {
        "data_root": str(data_root),
        "requested_sequences": requested_sequences,
        "requested_cameras": requested_cameras,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "assigned_sequence_camera_pairs": [
            {"sequence": pair.sequence, "camera": pair.camera} for pair in assigned_pairs
        ],
        "overwrite": overwrite,
        "continue_on_error": continue_on_error,
        "sequence_camera_runs": [
            {
                "sequence": item.sequence,
                "camera": item.camera,
                "total_images": item.total_images,
                "processed_images": item.processed_images,
                "skipped_images": item.skipped_images,
                "failed_images": item.failed_images,
            }
            for item in camera_stats
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
        "[panoptic-sam3-mask] completed "
        f"shard={summary['shard_index']}/{summary['num_shards']} "
        f"total_images={totals['total_images']} "
        f"processed={totals['processed_images']} "
        f"skipped={totals['skipped_images']} "
        f"failed={totals['failed_images']}"
    )
    if summary["failures"]:
        print("[panoptic-sam3-mask] failures:")
        for item in summary["failures"]:
            print(
                "[panoptic-sam3-mask] "
                f"{item['sequence']}:{item['camera']}:{item['image_path']}: {item['reason']}"
            )


def main() -> None:
    args = parse_args()
    requested_sequences = _parse_csv_arg(args.sequences)
    requested_cameras = _parse_csv_arg(args.cameras)
    _validate_startup(args)
    all_pairs = _build_sequence_camera_pairs(
        args.data_root,
        requested_sequences,
        requested_cameras,
    )
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
        f"[panoptic-sam3-mask] shard {args.shard_index}/{args.num_shards}: "
        f"{len(assigned_pairs)} sequence-camera pair(s) assigned"
    )
    segmentor = _load_human_segmentor(args.segmentor_path.expanduser(), args.device)

    camera_stats: list[CameraRunStats] = []
    failures: list[dict] = []
    for pair in assigned_pairs:
        sequence_name = pair.sequence
        camera_name = pair.camera
        sequence_root = _resolve_sequence_root(args.data_root, sequence_name)
        try:
            if not sequence_root.is_dir():
                raise FileNotFoundError(f"missing sequence directory: {sequence_root}")
            stats = _handle_sequence_camera(
                segmentor,
                sequence_root,
                sequence_name,
                camera_name,
                overwrite=args.overwrite,
                continue_on_error=args.continue_on_error,
                failures=failures,
            )
            camera_stats.append(stats)
        except Exception as exc:
            _record_failure(
                failures,
                sequence=sequence_name,
                camera=camera_name,
                image_path=None,
                reason=str(exc),
            )
            if not args.continue_on_error:
                raise

    summary = _build_summary(
        data_root=args.data_root,
        requested_sequences=requested_sequences,
        requested_cameras=requested_cameras,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        assigned_pairs=assigned_pairs,
        overwrite=args.overwrite,
        continue_on_error=args.continue_on_error,
        camera_stats=camera_stats,
        failures=failures,
    )
    _print_summary(summary)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        _fail(str(exc))
