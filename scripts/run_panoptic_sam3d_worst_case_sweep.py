#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.sam3d_panoptic_segment_eval.dataset_adapter import build_segments, load_dataset_context  # noqa: E402
from tools.sam3d_panoptic_segment_eval.reporting import compute_overall_metrics  # noqa: E402
from tools.sam3d_panoptic_segment_eval.run_segment_eval import _evaluate_one_frame, _load_estimator  # noqa: E402
from tools.sam3d_panoptic_segment_eval.run_utils import dump_json  # noqa: E402
from tools.sam3d_panoptic_segment_eval.segment_metrics import aggregate_segment_metrics  # noqa: E402
from tools.sam3d_panoptic_segment_eval.joint_adapter import SAM3ToPanopticCOCO19Adapter  # noqa: E402


DEFAULT_CONFIGS = [
    "configs/analysis/panoptic_sam3d_segment_eval_office1.yml",
    "configs/analysis/panoptic_sam3d_segment_eval_office2.yml",
    "configs/analysis/panoptic_sam3d_segment_eval_cello3.yml",
]
DEFAULT_CAMERAS = [f"kinect_{idx:03d}" for idx in range(1, 11)]
METRICS = ("mpjpe", "pa_mpjpe", "pc_mpjpe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SAM3D across all cameras for office1/office2/cello3, rank worst "
            "sequence-camera pairs and segments by metric, and export rerun recordings."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--configs",
        default=",".join(DEFAULT_CONFIGS),
        help="Comma-separated analysis config paths. Defaults to office1/office2/cello3 configs.",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--segment-length", required=True, type=int, help="Segment length for evaluation")
    parser.add_argument(
        "--cameras",
        default=",".join(DEFAULT_CAMERAS),
        help="Comma-separated camera names. Defaults to kinect_001..kinect_010.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-K worst items to keep per metric")
    parser.add_argument(
        "--checkpoint-root",
        default="/opt/data/SAM_3dbody_checkpoints",
        help="SAM-3D-Body checkpoint root",
    )
    parser.add_argument("--device", default="cuda", help="Inference device")
    parser.add_argument(
        "--segmentor-name",
        default="none",
        choices=["none", "sam2", "sam3"],
        help="Optional human segmentor backend for SAM3D",
    )
    parser.add_argument(
        "--segmentor-path",
        default="/opt/data/sam3_checkpoints",
        help="Segmentor checkpoint path",
    )
    parser.add_argument("--use-mask", action="store_true", help="Enable mask-conditioned SAM3D inference")
    parser.add_argument(
        "--invalid-frame-mode",
        default="drop",
        choices=["drop", "error"],
        help="How to handle invalid frames during evaluation",
    )
    parser.add_argument(
        "--output-root",
        default="logs/panoptic_sam3d_worst_case_sweep",
        help="Root directory for aggregate outputs",
    )
    parser.add_argument("--run-name", default=None, help="Optional run directory name override")
    parser.add_argument(
        "--max-segments-per-pair",
        type=int,
        default=None,
        help="Optional debug cap on segments per pair",
    )
    return parser.parse_args()


def _parse_csv_arg(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _sanitize_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(text))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _flatten_pair_frame_metrics(segment_rows: list[dict]) -> list[dict]:
    rows = []
    for segment in sorted(segment_rows, key=lambda row: int(row["segment_index"])):
        rows.extend(segment.get("frame_metrics", []))
    return rows


def _pair_sample_indices(segment_rows: list[dict]) -> tuple[list[int], list[int]]:
    sample_indices = []
    frame_ids = []
    for segment in sorted(segment_rows, key=lambda row: int(row["segment_index"])):
        sample_indices.extend(int(x) for x in segment["sample_indices"])
        frame_ids.extend(int(x) for x in segment["frame_ids"])
    return sample_indices, frame_ids


def _evaluate_pair(
    *,
    cfg_path: str,
    split: str,
    camera: str,
    segment_length: int,
    estimator,
    joint_adapter,
    use_mask: bool,
    invalid_frame_mode: str,
    max_segments_per_pair: int | None,
) -> tuple[dict, list[dict]]:
    dataset_ctx = load_dataset_context(cfg_path, split, camera)
    segments = build_segments(
        dataset_ctx,
        segment_length=segment_length,
        max_segments=max_segments_per_pair,
    )
    segment_rows: list[dict] = []
    for segment in segments:
        frame_results = []
        for sample_index, frame_id in zip(segment.sample_indices, segment.frame_ids, strict=True):
            frame_metric, _, _, _ = _evaluate_one_frame(
                dataset_ctx=dataset_ctx,
                sample_index=int(sample_index),
                frame_id=int(frame_id),
                estimator=estimator,
                joint_adapter=joint_adapter,
                use_mask=use_mask,
                invalid_frame_mode=invalid_frame_mode,
            )
            frame_results.append(frame_metric)

        segment_row = aggregate_segment_metrics(
            sequence_name=segment.sequence_name,
            camera_name=segment.camera_name,
            segment_index=segment.segment_index,
            segment_length=segment.segment_length,
            start_frame_id=segment.start_frame_id,
            end_frame_id=segment.end_frame_id,
            sample_indices=list(segment.sample_indices),
            frame_results=frame_results,
        )
        segment_row["cfg_path"] = str(Path(cfg_path).expanduser().resolve())
        segment_rows.append(segment_row)

    overall_metrics = compute_overall_metrics(segment_rows)
    sample_indices, frame_ids = _pair_sample_indices(segment_rows)
    pair_row = {
        "cfg_path": str(Path(cfg_path).expanduser().resolve()),
        "sequence_name": segments[0].sequence_name,
        "camera_name": camera,
        "segment_length": int(segment_length),
        "sample_indices": sample_indices,
        "frame_ids": frame_ids,
        "frame_metrics": _flatten_pair_frame_metrics(segment_rows),
        **overall_metrics,
    }
    return pair_row, segment_rows


def _rank_rows(rows: list[dict], metric_key: str, top_k: int) -> list[dict]:
    ranked = sorted(rows, key=lambda row: float(row[metric_key]), reverse=True)
    return ranked[:top_k]


def _make_run_dir(output_root: str, run_name: str | None, segment_length: int) -> Path:
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if run_name is None:
        run_name = f"seg{segment_length}_{int(time.time())}"
    run_dir = root / _sanitize_name(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_summary_files(run_dir: Path, pair_rows: list[dict], segment_rows: list[dict]) -> None:
    pair_fieldnames = [
        "sequence_name",
        "camera_name",
        "segment_length",
        "num_segments",
        "num_total_frames",
        "num_valid_frames",
        "num_invalid_frames",
        "mpjpe_mean",
        "pa_mpjpe_mean",
        "pc_mpjpe_mean",
        "mpjpe_max",
        "pa_mpjpe_max",
        "pc_mpjpe_max",
        "cfg_path",
    ]
    segment_fieldnames = [
        "sequence_name",
        "camera_name",
        "segment_index",
        "segment_length",
        "start_frame_id",
        "end_frame_id",
        "num_frames",
        "num_valid_frames",
        "num_invalid_frames",
        "mpjpe_mean",
        "pa_mpjpe_mean",
        "pc_mpjpe_mean",
        "mpjpe_max",
        "pa_mpjpe_max",
        "pc_mpjpe_max",
        "cfg_path",
    ]
    _write_csv(run_dir / "all_pairs.csv", pair_rows, pair_fieldnames)
    dump_json(run_dir / "all_pairs.json", pair_rows)
    _write_csv(run_dir / "all_segments.csv", segment_rows, segment_fieldnames)
    dump_json(run_dir / "all_segments.json", segment_rows)


def _write_topk_files(run_dir: Path, prefix: str, rows: list[dict], metric: str, fieldnames: list[str]) -> None:
    base = f"{prefix}_by_{metric}"
    _write_csv(run_dir / f"{base}.csv", rows, fieldnames)
    dump_json(run_dir / f"{base}.json", rows)


def _build_visualization_spec(
    *,
    item_type: str,
    metric: str,
    rank: int,
    row: dict,
    run_dir: Path,
) -> dict:
    title = f"{item_type} rank{rank} by {metric}: {row['sequence_name']} {row['camera_name']}"
    base_name = (
        f"{item_type}_{metric}_rank{rank}_{row['sequence_name']}_{row['camera_name']}"
    )
    if item_type == "segment":
        base_name += f"_seg{int(row['segment_index']):04d}"
    return {
        "title": title,
        "item_type": item_type,
        "metric_name": metric,
        "rank": rank,
        "cfg_path": row["cfg_path"],
        "split": args.split,
        "camera": row["camera_name"],
        "sequence_name": row["sequence_name"],
        "sample_indices": [int(x) for x in row["sample_indices"]],
        "frame_ids": [int(x) for x in row["frame_ids"]],
        "frame_metrics": row.get("frame_metrics", []),
        "recording_name": base_name,
        "out_rrd": str((run_dir / "rerun" / item_type / metric / f"{base_name}.rrd").resolve()),
    }


def _export_rerun_specs(run_dir: Path, specs: list[dict], args: argparse.Namespace) -> None:
    helper = REPO_ROOT / "scripts" / "export_panoptic_sam3d_overlay_rerun.py"
    specs_dir = run_dir / "rerun_specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        spec_path = specs_dir / f"{spec['recording_name']}.json"
        dump_json(spec_path, spec)
        cmd = [
            sys.executable,
            str(helper),
            "--spec",
            str(spec_path),
            "--checkpoint-root",
            args.checkpoint_root,
            "--device",
            args.device,
            "--segmentor-name",
            args.segmentor_name,
            "--segmentor-path",
            args.segmentor_path,
            "--invalid-frame-mode",
            args.invalid_frame_mode,
        ]
        if args.use_mask:
            cmd.append("--use-mask")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError(f"--top-k must be > 0, got {args.top_k}")

    cfg_paths = _parse_csv_arg(args.configs)
    cameras = _parse_csv_arg(args.cameras)
    run_dir = _make_run_dir(args.output_root, args.run_name, args.segment_length)

    estimator = _load_estimator(
        Path(args.checkpoint_root).expanduser().resolve(),
        args.device,
        args.segmentor_name,
        args.segmentor_path,
    )
    joint_adapter = SAM3ToPanopticCOCO19Adapter()

    pair_rows: list[dict] = []
    segment_rows_all: list[dict] = []
    skipped_pairs: list[dict] = []

    pair_jobs = [(cfg_path, camera) for cfg_path in cfg_paths for camera in cameras]
    progress = tqdm(pair_jobs, desc="Panoptic pair eval", unit="pair", dynamic_ncols=True)
    for cfg_path, camera in progress:
        progress.set_postfix(cfg=Path(cfg_path).stem, camera=camera)
        try:
            pair_row, segment_rows = _evaluate_pair(
                cfg_path=cfg_path,
                split=args.split,
                camera=camera,
                segment_length=args.segment_length,
                estimator=estimator,
                joint_adapter=joint_adapter,
                use_mask=args.use_mask,
                invalid_frame_mode=args.invalid_frame_mode,
                max_segments_per_pair=args.max_segments_per_pair,
            )
        except Exception as exc:  # noqa: BLE001
            skipped_pairs.append({"cfg_path": cfg_path, "camera": camera, "error": str(exc)})
            continue
        pair_rows.append(pair_row)
        segment_rows_all.extend(segment_rows)

    if not pair_rows:
        raise RuntimeError("No sequence-camera pairs were evaluated successfully.")

    _write_summary_files(run_dir, pair_rows, segment_rows_all)
    dump_json(run_dir / "skipped_pairs.json", skipped_pairs)

    pair_fieldnames = [
        "sequence_name",
        "camera_name",
        "segment_length",
        "num_segments",
        "num_total_frames",
        "num_valid_frames",
        "num_invalid_frames",
        "mpjpe_mean",
        "pa_mpjpe_mean",
        "pc_mpjpe_mean",
        "mpjpe_max",
        "pa_mpjpe_max",
        "pc_mpjpe_max",
        "cfg_path",
    ]
    segment_fieldnames = [
        "sequence_name",
        "camera_name",
        "segment_index",
        "segment_length",
        "start_frame_id",
        "end_frame_id",
        "num_frames",
        "num_valid_frames",
        "num_invalid_frames",
        "mpjpe_mean",
        "pa_mpjpe_mean",
        "pc_mpjpe_mean",
        "mpjpe_max",
        "pa_mpjpe_max",
        "pc_mpjpe_max",
        "cfg_path",
    ]

    rerun_specs: list[dict] = []
    topk_summary: dict[str, dict[str, list[dict]]] = {"pairs": {}, "segments": {}}
    for metric in METRICS:
        metric_key = f"{metric}_mean"
        top_pairs = _rank_rows(pair_rows, metric_key, args.top_k)
        top_segments = _rank_rows(segment_rows_all, metric_key, args.top_k)
        topk_summary["pairs"][metric] = top_pairs
        topk_summary["segments"][metric] = top_segments
        _write_topk_files(run_dir, "top_pairs", top_pairs, metric, pair_fieldnames)
        _write_topk_files(run_dir, "top_segments", top_segments, metric, segment_fieldnames)
        for rank, row in enumerate(top_pairs, start=1):
            rerun_specs.append(
                _build_visualization_spec(
                    item_type="pair",
                    metric=metric,
                    rank=rank,
                    row=row,
                    run_dir=run_dir,
                )
            )
        for rank, row in enumerate(top_segments, start=1):
            rerun_specs.append(
                _build_visualization_spec(
                    item_type="segment",
                    metric=metric,
                    rank=rank,
                    row=row,
                    run_dir=run_dir,
                )
            )

    dump_json(run_dir / "topk_summary.json", topk_summary)
    _export_rerun_specs(run_dir, rerun_specs, args)

    run_summary = {
        "configs": [str(Path(x).expanduser().resolve()) for x in cfg_paths],
        "split": args.split,
        "segment_length": int(args.segment_length),
        "cameras": cameras,
        "top_k": int(args.top_k),
        "num_evaluated_pairs": len(pair_rows),
        "num_skipped_pairs": len(skipped_pairs),
        "num_total_segments": len(segment_rows_all),
        "num_rerun_specs": len(rerun_specs),
        "run_dir": str(run_dir),
    }
    dump_json(run_dir / "run_summary.json", run_summary)
