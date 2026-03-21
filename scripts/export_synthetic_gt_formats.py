#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.synthetic_data.export_pipeline import (  # noqa: E402
    SyntheticTargetExportConfig,
    SyntheticTargetExportPipeline,
)
from projects.synthetic_data.mhr_smpl_adapter import MHRSMPLFitConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export HuMMan/Panoptic-compatible GT bundles for existing synthetic samples."
    )
    parser.add_argument("--synthetic-root", type=str, required=True)
    parser.add_argument("--checkpoint-root", type=str, default="/opt/data/SAM_3dbody_checkpoints")
    parser.add_argument("--mhr-repo-root", type=str, default=None)
    parser.add_argument("--smpl-model-path", type=str, default="weights/smpl/SMPL_NEUTRAL.pkl")
    parser.add_argument("--sample-dir", type=str, default=None, help="Process one explicit sample directory.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means process all remaining samples.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lidar-version", type=str, default="v0a")
    parser.add_argument("--no-pose-encodings", action="store_true")
    parser.add_argument("--no-rgb-2d", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SyntheticTargetExportConfig(
        synthetic_root=args.synthetic_root,
        checkpoint_root=args.checkpoint_root,
        device=args.device,
        lidar_version=args.lidar_version,
        save_pose_encodings=not args.no_pose_encodings,
        save_rgb_2d_keypoints=not args.no_rgb_2d,
        mhr_smpl=MHRSMPLFitConfig(
            mhr_repo_root=args.mhr_repo_root,
            smpl_model_path=args.smpl_model_path,
            device=args.device,
        ),
    )
    pipeline = SyntheticTargetExportPipeline(repo_root=REPO_ROOT, cfg=cfg)

    if args.sample_dir:
        result = pipeline.export_sample(Path(args.sample_dir))
        print(json.dumps(result, indent=2))
        return

    sample_dirs = pipeline.list_sample_dirs()
    start = max(0, args.start_index)
    if args.max_samples <= 0:
        end = len(sample_dirs)
    else:
        end = min(len(sample_dirs), start + int(args.max_samples))
    if start >= end:
        raise ValueError(f"Empty processing range: start={start}, end={end}, total={len(sample_dirs)}")

    results = []
    root_name = Path(args.synthetic_root).expanduser().resolve().name
    run_jsonl = Path(args.synthetic_root).expanduser().resolve() / f"export_results_{root_name}_{start}_{end}.jsonl"
    run_summary = Path(args.synthetic_root).expanduser().resolve() / f"export_summary_{root_name}_{start}_{end}.json"
    run_jsonl.parent.mkdir(parents=True, exist_ok=True)
    run_jsonl.write_text("", encoding="utf-8")
    for idx, sample_dir in enumerate(sample_dirs[start:end], start=start):
        manifest_path = sample_dir / cfg.export_dirname / "export_manifest.json"
        if args.resume and manifest_path.is_file():
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
            results.append(cached)
            with run_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(cached, ensure_ascii=True) + "\n")
            print(f"[{idx + 1}/{end}] skip existing: {sample_dir.name} status={cached.get('status')}")
            continue
        result = pipeline.export_sample(sample_dir)
        results.append(result)
        with run_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=True) + "\n")
        print(f"[{idx + 1}/{end}] {result['status']}: {sample_dir.name}")

    summary = {
        "synthetic_root": str(Path(args.synthetic_root).expanduser().resolve()),
        "start_index": start,
        "end_index": end,
        "processed": len(results),
        "accepted": sum(1 for item in results if item.get("status") == "accepted"),
        "rejected": sum(1 for item in results if item.get("status") != "accepted"),
        "results_jsonl": str(run_jsonl),
        "summary_json": str(run_summary),
        "results": [
            {
                "sample_dir": item.get("sample_dir"),
                "status": item.get("status"),
                "rejection_reason": item.get("rejection_reason"),
                "manifest_path": item.get("manifest_path"),
            }
            for item in results
        ],
    }
    run_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
