#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.synthetic_data import SyntheticGenerationConfig, SyntheticGenerationPipeline
from projects.synthetic_data.virtual_lidar import VirtualLidarConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate v0-a synthetic samples for a full COCO split using SAM-3D-Body."
    )
    parser.add_argument("--data-root", type=str, default="/opt/data/coco")
    parser.add_argument(
        "--split",
        type=str,
        default="val2017",
        choices=["val2017", "train2017"],
        help="COCO split to process.",
    )
    parser.add_argument("--checkpoint-root", type=str, default="/opt/data/SAM_3dbody_checkpoints")
    parser.add_argument("--output-dir", type=str, default="logs/synthetic_data/v0a_dataset")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--end-index",
        type=int,
        default=-1,
        help="Inclusive-exclusive stop index over eligible samples. -1 means process to the end of the split.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Optional cap on processed eligible samples from start-index. -1 means no cap.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip indices whose manifest.json already exists.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-area", type=float, default=4096.0)
    parser.add_argument("--min-keypoints", type=int, default=5)
    parser.add_argument("--min-mask-pixels", type=int, default=1024)
    parser.add_argument("--allow-multi-person-images", action="store_true")
    parser.add_argument("--lidar-radius-min", type=float, default=2.5)
    parser.add_argument("--lidar-radius-max", type=float, default=4.0)
    parser.add_argument("--lidar-elevation-min-deg", type=float, default=5.0)
    parser.add_argument("--lidar-elevation-max-deg", type=float, default=35.0)
    parser.add_argument("--lidar-num-points", type=int, default=2048)
    parser.add_argument("--lidar-oversample-factor", type=int, default=8)
    parser.add_argument("--lidar-surface-noise-std", type=float, default=0.002)
    parser.add_argument("--save-source-rgb", action="store_true")
    parser.add_argument("--save-visualizations", action="store_true")
    return parser.parse_args()


def _split_paths(split: str) -> tuple[str, str]:
    if split == "val2017":
        return "annotations/person_keypoints_val2017.json", "val2017"
    if split == "train2017":
        return "annotations/person_keypoints_train2017.json", "train2017"
    raise ValueError(f"Unsupported split={split}")


def _load_manifest(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    annotation_file, image_dir = _split_paths(args.split)
    cfg = SyntheticGenerationConfig(
        data_root=args.data_root,
        annotation_file=annotation_file,
        image_dir=image_dir,
        checkpoint_root=args.checkpoint_root,
        output_root=args.output_dir,
        min_area=args.min_area,
        min_keypoints=args.min_keypoints,
        one_person_only=not args.allow_multi_person_images,
        min_mask_pixels=args.min_mask_pixels,
        seed=args.seed,
        save_source_rgb=args.save_source_rgb,
        save_visualizations=args.save_visualizations,
        lidar=VirtualLidarConfig(
            radius_range=(args.lidar_radius_min, args.lidar_radius_max),
            elevation_range_deg=(args.lidar_elevation_min_deg, args.lidar_elevation_max_deg),
            num_points=args.lidar_num_points,
            oversample_factor=args.lidar_oversample_factor,
            surface_noise_std=args.lidar_surface_noise_std,
        ),
    )
    pipeline = SyntheticGenerationPipeline(repo_root=REPO_ROOT, cfg=cfg)

    dataset_size = len(pipeline)
    start_index = max(0, args.start_index)
    end_index = dataset_size if args.end_index < 0 else min(dataset_size, args.end_index)
    if start_index >= end_index:
        raise ValueError(
            f"Empty processing range: start_index={start_index}, end_index={end_index}, dataset_size={dataset_size}"
        )
    if args.max_samples > 0:
        end_index = min(end_index, start_index + args.max_samples)

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_summary_path = output_root / f"run_summary_{args.split}_{start_index}_{end_index}.json"
    run_results_path = output_root / f"run_results_{args.split}_{start_index}_{end_index}.jsonl"

    processed = 0
    accepted = 0
    rejected = 0
    skipped = 0
    results: list[dict] = []

    with run_results_path.open("a", encoding="utf-8") as results_file:
        for index in range(start_index, end_index):
            manifest_path = pipeline.manifest_path_for_index(index)
            if args.resume and manifest_path.is_file():
                cached = _load_manifest(manifest_path)
                if cached is not None:
                    status = cached.get("status", "unknown")
                    skipped += 1
                    results.append(
                        {
                            "annotation_index": index,
                            "annotation_id": cached.get("annotation_id"),
                            "image_id": cached.get("image_id"),
                            "status": status,
                            "rejection_reason": cached.get("rejection_reason"),
                            "manifest_path": str(manifest_path),
                            "skipped_existing": True,
                        }
                    )
                    if status == "accepted":
                        accepted += 1
                    else:
                        rejected += 1
                    print(
                        f"[{index + 1}/{end_index}] skip existing: "
                        f"annotation_id={cached.get('annotation_id')} status={status}"
                    )
                    continue

            item = pipeline.process_index(index)
            processed += 1
            if item["status"] == "accepted":
                accepted += 1
            else:
                rejected += 1
            results.append(
                {
                    "annotation_index": item["annotation_index"],
                    "annotation_id": item["annotation_id"],
                    "image_id": item["image_id"],
                    "status": item["status"],
                    "rejection_reason": item["rejection_reason"],
                    "manifest_path": item["manifest_path"],
                    "skipped_existing": False,
                }
            )
            results_file.write(json.dumps(results[-1]) + "\n")
            results_file.flush()
            print(
                f"[{index + 1}/{end_index}] {item['status']}: "
                f"annotation_id={item['annotation_id']} image_id={item['image_id']}"
            )

    summary = {
        "split": args.split,
        "dataset_size": dataset_size,
        "start_index": start_index,
        "end_index": end_index,
        "processed": processed,
        "accepted": accepted,
        "rejected": rejected,
        "skipped_existing": skipped,
        "output_root": str(output_root),
        "results_file": str(run_results_path),
        "results": results,
    }
    run_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
