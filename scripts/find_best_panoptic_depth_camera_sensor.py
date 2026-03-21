#!/usr/bin/env python3
import argparse
import csv
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.mpjpe import mpjpe_func, pampjpe_func
from tools.eval_per_frame_sensor import (
    _build_seq_group_names,
    _parse_sensor_index_map,
    _resolve_camera_key,
    _run_sample_worker_pool,
)


_SAMPLE_RE = re.compile(
    r"^(?P<seq>\d{6}_[A-Za-z0-9]+)_rgb_kinect_(?P<rgb>\d{3})_depth_kinect_(?P<depth>\d{3})_(?P<frame>\d+)$"
)


def _parse_rgb_list(text: str | None) -> list[str]:
    if not text:
        return []
    out = []
    for item in text.split(","):
        token = item.strip()
        if not token:
            continue
        if token.startswith("kinect_"):
            token = token.split("_")[-1]
        out.append(f"{int(token):03d}")
    return out


def _parse_pair(sample_id: str):
    match = _SAMPLE_RE.match(sample_id)
    if match is None:
        return None
    return (
        match.group("seq"),
        match.group("rgb"),
        match.group("depth"),
        int(match.group("frame")),
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank Panoptic depth cameras by sensor-space MPJPE from a saved prediction pickle."
    )
    parser.add_argument("--pred-file", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--rgb-cameras", default="")
    parser.add_argument("--pred-keypoints-key", default="pred_keypoints")
    parser.add_argument("--gt-keypoints-key", default="gt_keypoints")
    parser.add_argument("--pred-cameras-key", default="pred_cameras_stream")
    parser.add_argument("--gt-cameras-key", default="gt_cameras_stream")
    parser.add_argument("--pose-encoding-type", default="absT_quaR_FoV")
    parser.add_argument("--target-modality", default="lidar")
    parser.add_argument("--sensor-index-by-modality", default="")
    parser.add_argument("--lidar-modality-index", type=int, default=None)
    parser.add_argument("--rgb-modality-index", type=int, default=None)
    parser.add_argument("--pelvis-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-csv", default="")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    with open(args.pred_file, "rb") as f:
        data = pickle.load(f)

    pred_camera_key = _resolve_camera_key(
        data=data,
        requested_key=args.pred_cameras_key,
        fallback_keys=["pred_cameras_stream", "pred_cameras"],
    )
    gt_camera_key = _resolve_camera_key(
        data=data,
        requested_key=args.gt_cameras_key,
        fallback_keys=["gt_cameras_stream", "gt_cameras"],
    )

    pred_all = data[args.pred_keypoints_key]
    gt_all = data[args.gt_keypoints_key]
    sample_ids = data.get("sample_ids") or [f"sample_{i}" for i in range(len(pred_all))]
    seq_group_names = _build_seq_group_names(data, sample_ids)

    target_modality = str(args.target_modality).lower()
    sensor_index_by_modality = _parse_sensor_index_map(args.sensor_index_by_modality)
    if target_modality not in sensor_index_by_modality:
        sensor_index_by_modality[target_modality] = 0
    fallback_idx = {
        "lidar": args.lidar_modality_index,
        "rgb": args.rgb_modality_index,
    }.get(target_modality)
    target_sensor_idx = int(sensor_index_by_modality[target_modality])
    worker_context = {
        "data": data,
        "pred_all": pred_all,
        "gt_all": gt_all,
        "sample_ids": sample_ids,
        "seq_group_names": seq_group_names,
        "pred_camera_key": pred_camera_key,
        "gt_camera_key": gt_camera_key,
        "target_modality": target_modality,
        "fallback_idx": fallback_idx,
        "target_sensor_idx": target_sensor_idx,
        "pose_encoding_type": args.pose_encoding_type,
    }

    allowed_rgbs = set(_parse_rgb_list(args.rgb_cameras))
    grouped_pred = defaultdict(list)
    grouped_gt = defaultdict(list)
    grouped_frames = defaultdict(list)
    skipped_parse = 0

    for status, sample_idx, _seq_name, frame_idx, pred_proj, gt_proj in _run_sample_worker_pool(
        context=worker_context,
        num_samples=len(sample_ids),
        workers=int(args.workers),
        show_progress=True,
        desc="project per-frame sensor",
    ):
        if status != "ok":
            continue
        parsed = _parse_pair(sample_ids[sample_idx])
        if parsed is None:
            skipped_parse += 1
            continue
        seq, rgb, depth, frame = parsed
        if seq != args.sequence:
            continue
        if allowed_rgbs and rgb not in allowed_rgbs:
            continue
        key = (seq, rgb, depth)
        grouped_pred[key].append(pred_proj)
        grouped_gt[key].append(gt_proj)
        grouped_frames[key].append(frame_idx if frame_idx is not None else frame)

    pelvis = int(args.pelvis_index)
    results = []
    for (seq, rgb, depth), pred_list in grouped_pred.items():
        pred = np.stack(pred_list, axis=0)
        gt = np.stack(grouped_gt[(seq, rgb, depth)], axis=0)
        frames = sorted(int(x) for x in grouped_frames[(seq, rgb, depth)])
        pred_centered = pred - pred[:, pelvis : pelvis + 1, :]
        gt_centered = gt - gt[:, pelvis : pelvis + 1, :]
        results.append(
            {
                "sequence": seq,
                "rgb_camera": f"kinect_{rgb}",
                "depth_camera": f"kinect_{depth}",
                "sample_count": int(pred.shape[0]),
                "frame_start": int(frames[0]),
                "frame_end": int(frames[-1]),
                "sensor_mpjpe_m": float(mpjpe_func(pred, gt, reduce=True)),
                "sensor_pampjpe_m": float(pampjpe_func(pred, gt, reduce=True)),
                "sensor_centered_mpjpe_m": float(mpjpe_func(pred_centered, gt_centered, reduce=True)),
            }
        )

    results.sort(key=lambda row: (row["rgb_camera"], row["sensor_mpjpe_m"], row["depth_camera"]))
    if not results:
        raise RuntimeError("No sequence/rgb/depth pairs matched the requested filters.")

    printed = 0
    for row in results:
        if not allowed_rgbs and printed >= int(args.top_k):
            break
        print(
            f"{row['sequence']} rgb={row['rgb_camera']} depth={row['depth_camera']} "
            f"mpjpe={row['sensor_mpjpe_m']:.6f}m pa={row['sensor_pampjpe_m']:.6f}m "
            f"centered={row['sensor_centered_mpjpe_m']:.6f}m n={row['sample_count']} "
            f"frames={row['frame_start']}..{row['frame_end']}"
        )
        printed += 1

    summary = {
        "pred_file": str(Path(args.pred_file).resolve()),
        "sequence": args.sequence,
        "target_modality": target_modality,
        "target_sensor_idx": target_sensor_idx,
        "workers": int(args.workers),
        "skipped_unparseable_sample_ids": int(skipped_parse),
        "results": results,
    }

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2))
    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sequence",
                    "rgb_camera",
                    "depth_camera",
                    "sample_count",
                    "frame_start",
                    "frame_end",
                    "sensor_mpjpe_m",
                    "sensor_pampjpe_m",
                    "sensor_centered_mpjpe_m",
                ],
            )
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main()
