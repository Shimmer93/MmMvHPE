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
        description="Generate v0-a synthetic RGB-to-LiDAR samples from COCO val using SAM-3D-Body."
    )
    parser.add_argument("--data-root", type=str, default="/opt/data/coco")
    parser.add_argument(
        "--annotation-file",
        type=str,
        default="annotations/person_keypoints_val2017.json",
        help="COCO annotation file relative to --data-root.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="val2017",
        help="COCO image directory relative to --data-root.",
    )
    parser.add_argument("--checkpoint-root", type=str, default="/opt/data/SAM_3dbody_checkpoints")
    parser.add_argument("--output-dir", type=str, default="logs/synthetic_data/v0a")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=1)
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


def main() -> None:
    args = parse_args()
    cfg = SyntheticGenerationConfig(
        data_root=args.data_root,
        annotation_file=args.annotation_file,
        image_dir=args.image_dir,
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
    results = pipeline.process_range(start_index=args.start_index, max_samples=args.max_samples)

    accepted = [item for item in results if item["status"] == "accepted"]
    rejected = [item for item in results if item["status"] != "accepted"]
    summary = {
        "total": len(results),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "output_root": str(Path(args.output_dir).expanduser().resolve()),
        "results": [
            {
                "annotation_index": item["annotation_index"],
                "annotation_id": item["annotation_id"],
                "image_id": item["image_id"],
                "status": item["status"],
                "rejection_reason": item["rejection_reason"],
                "manifest_path": item["manifest_path"],
            }
            for item in results
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
