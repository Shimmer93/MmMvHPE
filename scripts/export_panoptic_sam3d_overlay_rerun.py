#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import rerun as rr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.sam3d_panoptic_segment_eval.run_segment_eval import (  # noqa: E402
    _blend_overlays,
    _evaluate_one_frame,
    _load_estimator,
)
from tools.sam3d_panoptic_segment_eval.dataset_adapter import load_dataset_context  # noqa: E402
from tools.sam3d_panoptic_segment_eval.joint_adapter import SAM3ToPanopticCOCO19Adapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one Panoptic SAM3D overlay rerun recording from a JSON spec.",
        allow_abbrev=False,
    )
    parser.add_argument("--spec", required=True, help="Path to visualization spec JSON.")
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
        help="How to handle invalid frames during rerun export",
    )
    return parser.parse_args()


def _load_spec(spec_path: str) -> dict:
    path = Path(spec_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Spec file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    spec = _load_spec(args.spec)

    dataset_ctx = load_dataset_context(spec["cfg_path"], spec["split"], spec["camera"])
    estimator = _load_estimator(
        Path(args.checkpoint_root).expanduser().resolve(),
        args.device,
        args.segmentor_name,
        args.segmentor_path,
    )
    joint_adapter = SAM3ToPanopticCOCO19Adapter()

    out_rrd = Path(spec["out_rrd"]).expanduser().resolve()
    out_rrd.parent.mkdir(parents=True, exist_ok=True)

    rr.init(str(spec.get("recording_name", out_rrd.stem)), spawn=False)
    rr.save(str(out_rrd))
    rr.log("world/info/title", rr.TextLog(str(spec.get("title", out_rrd.stem))), static=True)
    rr.log("world/info/item_type", rr.TextLog(str(spec.get("item_type", "unknown"))), static=True)
    rr.log("world/info/metric_name", rr.TextLog(str(spec.get("metric_name", "unknown"))), static=True)
    rr.log("world/info/sequence_name", rr.TextLog(str(spec.get("sequence_name", "unknown"))), static=True)
    rr.log("world/info/camera_name", rr.TextLog(str(spec.get("camera", "unknown"))), static=True)

    per_frame_metrics = {
        (int(item["sample_index"]), int(item["frame_id"])): item
        for item in spec.get("frame_metrics", [])
    }

    sample_indices = [int(x) for x in spec["sample_indices"]]
    frame_ids = [int(x) for x in spec["frame_ids"]]
    for local_idx, (sample_index, frame_id) in enumerate(zip(sample_indices, frame_ids, strict=True)):
        frame_metric, rgb_image, gt_2d, pred_2d = _evaluate_one_frame(
            dataset_ctx=dataset_ctx,
            sample_index=sample_index,
            frame_id=frame_id,
            estimator=estimator,
            joint_adapter=joint_adapter,
            use_mask=args.use_mask,
            invalid_frame_mode=args.invalid_frame_mode,
        )
        overlay = rgb_image if gt_2d is None or pred_2d is None else _blend_overlays(rgb_image, gt_2d, pred_2d)

        rr.set_time("frame", sequence=local_idx)
        rr.log("world/info/sample_index", rr.TextLog(str(sample_index)))
        rr.log("world/info/frame_id", rr.TextLog(str(frame_id)))
        metric_row = per_frame_metrics.get((sample_index, frame_id))
        if metric_row is None:
            metric_row = {
                "mpjpe": frame_metric.mpjpe,
                "pa_mpjpe": frame_metric.pa_mpjpe,
                "pc_mpjpe": frame_metric.pc_mpjpe,
                "valid": frame_metric.valid,
                "reason": frame_metric.reason,
            }
        rr.log("world/info/frame_metrics", rr.TextLog(json.dumps(metric_row, sort_keys=True)))
        rr.log("world/rgb", rr.Image(rgb_image))
        rr.log("world/overlay", rr.Image(overlay))


if __name__ == "__main__":
    main()
