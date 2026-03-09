#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "sam-3d-body"))

from dataset_adapter import build_segments, load_dataset_context
from joint_adapter import PANOPTIC_DATASET19_BONES, SAM3ToPanopticCOCO19Adapter
from reporting import (
    compute_overall_metrics,
    group_by_sequence_camera,
    rank_worst_segments,
    write_overall_summary,
    write_grouped_summary,
    write_ranked_summary,
    write_segment_logs,
    write_static_plots,
)
from run_utils import dump_json, make_run_dir
from segment_metrics import (
    FrameMetricResult,
    aggregate_segment_metrics,
    evaluate_frame_metrics,
    invalid_frame_result,
)
from misc.vis import denormalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SAM-3D-Body on Panoptic COCO19 over fixed-length segments.",
        allow_abbrev=False,
    )
    parser.add_argument("--cfg", required=True, help="Config path")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--camera", required=True, help="Camera name, for example `kinect_008`")
    parser.add_argument("--segment-length", required=True, type=int, help="One fixed segment length per run")
    parser.add_argument(
        "--checkpoint-root",
        default="/opt/data/SAM_3dbody_checkpoints",
        help="SAM-3D-Body checkpoint root containing model_config.yaml, model.ckpt, and mhr_model.pt",
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
        help="Segmentor checkpoint path. For `sam3`, pass a `.pt` file or a directory containing `sam3.pt`.",
    )
    parser.add_argument("--use-mask", action="store_true", help="Enable mask-conditioned SAM3D inference")
    parser.add_argument(
        "--invalid-frame-mode",
        default="drop",
        choices=["drop", "error"],
        help="How to handle invalid frames such as 0 or >1 detections",
    )
    parser.add_argument(
        "--rank-metric",
        default="mpjpe",
        choices=["mpjpe", "pa_mpjpe", "pc_mpjpe"],
        help="Metric used to rank worst segments",
    )
    parser.add_argument(
        "--export-worst-k",
        type=int,
        default=0,
        help="Export visual inspection artifacts for the worst K segments",
    )
    parser.add_argument("--run-name", default=None, help="Optional run directory name override")
    parser.add_argument(
        "--output-root",
        default="logs/sam3d_panoptic_segment_eval",
        help="Root directory for logs and exported artifacts",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Optional debug cap on the number of scored segments",
    )
    return parser.parse_args()


def _check_checkpoint_paths(checkpoint_root: Path) -> tuple[Path, Path, Path]:
    if checkpoint_root.is_file():
        raise FileNotFoundError(
            f"`--checkpoint-root` must be a directory, got file: {checkpoint_root}"
        )
    cfg_path = checkpoint_root / "model_config.yaml"
    ckpt_path = checkpoint_root / "model.ckpt"
    mhr_path = checkpoint_root / "assets" / "mhr_model.pt"
    if not mhr_path.exists():
        mhr_path = checkpoint_root / "mhr_model.pt"
    missing = [str(path) for path in (cfg_path, ckpt_path, mhr_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required SAM-3D-Body files: " + ", ".join(missing))
    return cfg_path, ckpt_path, mhr_path


def _load_estimator(checkpoint_root: Path, device: str, segmentor_name: str, segmentor_path: str):
    _, ckpt_path, mhr_path = _check_checkpoint_paths(checkpoint_root)
    sam3d_root = REPO_ROOT / "third_party" / "sam-3d-body"
    if str(sam3d_root) not in sys.path:
        sys.path.insert(0, str(sam3d_root))
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    build_sam_path = sam3d_root / "tools" / "build_sam.py"
    if not build_sam_path.is_file():
        raise FileNotFoundError(f"SAM3D build_sam.py not found: {build_sam_path}")
    spec = importlib.util.spec_from_file_location("sam3d_build_sam", build_sam_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load SAM3D build_sam module from {build_sam_path}")
    build_sam_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_sam_module)
    HumanSegmentor = build_sam_module.HumanSegmentor

    model, model_cfg = load_sam_3d_body(
        checkpoint_path=str(ckpt_path),
        device=device,
        mhr_path=str(mhr_path),
    )
    human_segmentor = None
    if segmentor_name != "none":
        human_segmentor = HumanSegmentor(
            name=segmentor_name,
            device=device,
            path=segmentor_path,
        )
    return SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=human_segmentor,
        fov_estimator=None,
    )


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _process_image_for_display(image, denorm_params: dict | None, key: str) -> np.ndarray:
    arr = _to_numpy(image)
    if arr.ndim == 4:
        arr = arr[-1]
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
    if denorm_params is not None:
        if key == "rgb":
            mean = denorm_params.get("rgb_mean", [123.675, 116.28, 103.53])
            std = denorm_params.get("rgb_std", [58.395, 57.12, 57.375])
        else:
            mean = denorm_params.get("depth_mean", [0.0])
            std = denorm_params.get("depth_std", [255.0])
        arr = denormalize(arr, mean, std)
    if arr.dtype != np.uint8:
        arr = np.asarray(arr, dtype=np.float32)
        if np.nanmin(arr) >= 0.0 and np.nanmax(arr) <= 1.0 + 1e-6:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return arr


def _world_to_camera(points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    ext = np.asarray(extrinsic, dtype=np.float32)
    if ext.shape != (3, 4):
        raise ValueError(f"Expected camera extrinsic with shape (3,4), got {ext.shape}.")
    return (pts @ ext[:, :3].T + ext[:, 3]).astype(np.float32)


def _project_points(points_cam: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    points = np.asarray(points_cam, dtype=np.float32)
    K = np.asarray(intrinsic, dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"Expected intrinsic matrix shape (3,3), got {K.shape}.")
    z = np.clip(points[:, 2:3], 1e-6, None)
    xy = points[:, :2] / z
    uv = (xy @ K[:2, :2].T) + K[:2, 2]
    return uv.astype(np.float32)


def _draw_2d_skeleton(image_rgb: np.ndarray, keypoints_2d: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    canvas = image_rgb.copy()
    kpts = np.asarray(keypoints_2d, dtype=np.float32)
    for src_idx, dst_idx in PANOPTIC_DATASET19_BONES:
        p1 = kpts[src_idx]
        p2 = kpts[dst_idx]
        if np.isfinite(p1).all() and np.isfinite(p2).all():
            cv2.line(
                canvas,
                (int(round(p1[0])), int(round(p1[1]))),
                (int(round(p2[0])), int(round(p2[1]))),
                color,
                2,
                cv2.LINE_AA,
            )
    for x, y in kpts[:, :2]:
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(canvas, (int(round(x)), int(round(y))), 2, color, -1)
    return canvas


def _blend_overlays(rgb: np.ndarray, gt_2d: np.ndarray, pred_2d: np.ndarray) -> np.ndarray:
    canvas = _draw_2d_skeleton(rgb, gt_2d, color=(0, 255, 80))
    canvas = _draw_2d_skeleton(canvas, pred_2d, color=(255, 140, 0))
    return canvas


def _save_segment_metric_chart(segment_dir: Path, frame_exports: list[dict]) -> Path:
    frame_ids = [int(item["frame_id"]) for item in frame_exports]
    mpjpe = [float(item["frame_metrics"]["mpjpe"]) for item in frame_exports]
    pa_mpjpe = [float(item["frame_metrics"]["pa_mpjpe"]) for item in frame_exports]
    pc_mpjpe = [float(item["frame_metrics"]["pc_mpjpe"]) for item in frame_exports]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(frame_ids, mpjpe, marker="o", linewidth=2, color="#DD8452", label="MPJPE")
    ax.plot(frame_ids, pa_mpjpe, marker="o", linewidth=2, color="#4C72B0", label="PA-MPJPE")
    ax.plot(frame_ids, pc_mpjpe, marker="o", linewidth=2, color="#55A868", label="PC-MPJPE")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Meters")
    ax.set_title("Per-Frame Segment Metrics")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(frame_ids)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    out_path = segment_dir / "frame_metric_chart.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _evaluate_one_frame(
    *,
    dataset_ctx,
    sample_index: int,
    frame_id: int,
    estimator,
    joint_adapter,
    use_mask: bool,
    invalid_frame_mode: str,
) -> tuple[FrameMetricResult, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    sample = dataset_ctx.dataset[sample_index]
    selected_rgb = list(sample.get("selected_cameras", {}).get("rgb", []))
    if selected_rgb != [dataset_ctx.camera]:
        raise ValueError(
            f"Sample camera mismatch at sample_index={sample_index}: expected {[dataset_ctx.camera]}, got {selected_rgb}."
        )
    if "input_rgb" not in sample:
        raise ValueError(f"Sample idx {sample_index} does not contain `input_rgb`.")
    if "rgb_camera" not in sample:
        raise ValueError(f"Sample idx {sample_index} does not contain `rgb_camera`.")

    rgb_image = _process_image_for_display(sample["input_rgb"], dataset_ctx.denorm_params, key="rgb")
    outputs = estimator.process_one_image(rgb_image, use_mask=use_mask)
    if len(outputs) != 1:
        reason = f"expected_single_person_got_{len(outputs)}"
        if invalid_frame_mode == "error":
            raise RuntimeError(
                f"Invalid SAM3 output for sample_index={sample_index}, frame_id={frame_id}: {reason}"
            )
        frame_result = invalid_frame_result(sample_index=sample_index, frame_id=frame_id, reason=reason)
        return frame_result, rgb_image, None, None

    person = outputs[0]
    pred_joints_3d = np.asarray(person["pred_keypoints_3d"], dtype=np.float32)
    pred_cam_t = np.asarray(person["pred_cam_t"], dtype=np.float32).reshape(1, 3)
    pred_panoptic_cam = joint_adapter.adapt(pred_joints_3d + pred_cam_t)

    rgb_camera = sample["rgb_camera"]
    if isinstance(rgb_camera, list):
        if len(rgb_camera) != 1:
            raise ValueError(f"Expected one rgb_camera entry, got {len(rgb_camera)}.")
        rgb_camera = rgb_camera[0]
    gt_panoptic_world = _to_numpy(sample["gt_keypoints"]).astype(np.float32)
    gt_panoptic_cam = _world_to_camera(gt_panoptic_world, rgb_camera["extrinsic"])

    frame_metric = evaluate_frame_metrics(
        pred_panoptic_cam,
        gt_panoptic_cam,
        sample_index=sample_index,
        frame_id=frame_id,
        pelvis_idx=2,
    )

    pred_2d = _project_points(pred_panoptic_cam, rgb_camera["intrinsic"])
    gt_2d = _project_points(gt_panoptic_cam, rgb_camera["intrinsic"])
    return frame_metric, rgb_image, gt_2d, pred_2d


def _export_worst_segments(
    *,
    run_dir: Path,
    worst_rows: list[dict],
    dataset_ctx,
    estimator,
    joint_adapter,
    use_mask: bool,
    invalid_frame_mode: str,
    rank_metric: str,
) -> list[dict]:
    exports_root = run_dir / "worst_segment_exports"
    exports_root.mkdir(parents=True, exist_ok=True)
    export_metadata: list[dict] = []
    for rank_idx, row in enumerate(worst_rows, start=1):
        segment_dir = exports_root / (
            f"{rank_idx:03d}_{row['sequence_name']}_{row['camera_name']}_"
            f"seg{row['segment_index']:04d}_{rank_metric}"
        )
        segment_dir.mkdir(parents=True, exist_ok=True)
        frame_exports = []
        for frame_offset, (sample_index, frame_id) in enumerate(
            zip(row["sample_indices"], row["frame_ids"], strict=True)
        ):
            frame_result, rgb_image, gt_2d, pred_2d = _evaluate_one_frame(
                dataset_ctx=dataset_ctx,
                sample_index=int(sample_index),
                frame_id=int(frame_id),
                estimator=estimator,
                joint_adapter=joint_adapter,
                use_mask=use_mask,
                invalid_frame_mode=invalid_frame_mode,
            )
            if gt_2d is not None and pred_2d is not None:
                overlay = _blend_overlays(rgb_image, gt_2d, pred_2d)
            else:
                overlay = rgb_image
            out_path = segment_dir / f"{frame_offset:02d}_{int(frame_id):08d}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            frame_exports.append(
                {
                    "frame_offset": frame_offset,
                    "sample_index": int(sample_index),
                    "frame_id": int(frame_id),
                    "overlay_path": str(out_path.relative_to(run_dir)),
                    "frame_metrics": {
                        "sample_index": frame_result.sample_index,
                        "frame_id": frame_result.frame_id,
                        "valid": frame_result.valid,
                        "reason": frame_result.reason,
                        "mpjpe": frame_result.mpjpe,
                        "pa_mpjpe": frame_result.pa_mpjpe,
                        "pc_mpjpe": frame_result.pc_mpjpe,
                    },
                }
            )

        metadata = {
            "rank": rank_idx,
            "rank_metric": rank_metric,
            "sequence_name": row["sequence_name"],
            "camera_name": row["camera_name"],
            "segment_index": row["segment_index"],
            "segment_length": row["segment_length"],
            "start_frame_id": row["start_frame_id"],
            "end_frame_id": row["end_frame_id"],
            "sample_indices": row["sample_indices"],
            "frame_ids": row["frame_ids"],
            "segment_metrics": {
                "mpjpe_mean": row["mpjpe_mean"],
                "pa_mpjpe_mean": row["pa_mpjpe_mean"],
                "pc_mpjpe_mean": row["pc_mpjpe_mean"],
            },
            "frames": frame_exports,
        }
        chart_path = _save_segment_metric_chart(segment_dir, frame_exports)
        metadata["frame_metric_chart_path"] = str(chart_path.relative_to(run_dir))
        dump_json(segment_dir / "segment_metadata.json", metadata)
        export_metadata.append(metadata)

    dump_json(exports_root / "export_index.json", export_metadata)
    return export_metadata


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[sam3d-panoptic-segment-eval] CUDA not available, falling back to CPU.")
        args.device = "cpu"
    if args.use_mask and args.segmentor_name == "none":
        raise ValueError("`--use-mask` requires `--segmentor-name` to be `sam2` or `sam3`.")
    if args.export_worst_k < 0:
        raise ValueError(f"--export-worst-k must be >= 0, got {args.export_worst_k}.")

    start_time = time.time()
    dataset_ctx = load_dataset_context(args.cfg, args.split, args.camera)
    segments = build_segments(
        dataset_ctx,
        segment_length=args.segment_length,
        max_segments=args.max_segments,
    )
    run_dir = make_run_dir(
        cfg_path=args.cfg,
        split=args.split,
        camera=args.camera,
        segment_length=args.segment_length,
        output_root=args.output_root,
        run_name=args.run_name,
        segmentor_name=args.segmentor_name,
        use_mask=args.use_mask,
    )

    estimator = _load_estimator(
        Path(args.checkpoint_root).expanduser().resolve(),
        args.device,
        args.segmentor_name,
        args.segmentor_path,
    )
    joint_adapter = SAM3ToPanopticCOCO19Adapter()

    print(
        "[sam3d-panoptic-segment-eval] "
        f"segments={len(segments)} split={args.split} camera={args.camera} "
        f"segment_length={args.segment_length} run_dir={run_dir}"
    )

    segment_rows: list[dict] = []
    segment_progress = tqdm(segments, desc="Segments", unit="segment", dynamic_ncols=True)
    for segment in segment_progress:
        segment_progress.set_postfix(
            sequence=segment.sequence_name,
            start=segment.start_frame_id,
            end=segment.end_frame_id,
        )
        frame_results = []
        for sample_index, frame_id in zip(segment.sample_indices, segment.frame_ids, strict=True):
            frame_metric, _, _, _ = _evaluate_one_frame(
                dataset_ctx=dataset_ctx,
                sample_index=int(sample_index),
                frame_id=int(frame_id),
                estimator=estimator,
                joint_adapter=joint_adapter,
                use_mask=args.use_mask,
                invalid_frame_mode=args.invalid_frame_mode,
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
        segment_rows.append(segment_row)

    grouped_rows = group_by_sequence_camera(segment_rows)
    ranked_rows = rank_worst_segments(segment_rows, args.rank_metric)
    overall_metrics = compute_overall_metrics(segment_rows)

    write_segment_logs(run_dir, segment_rows)
    write_grouped_summary(run_dir, grouped_rows)
    write_overall_summary(run_dir, overall_metrics)
    write_ranked_summary(run_dir, ranked_rows, args.rank_metric)
    write_static_plots(run_dir, segment_rows, grouped_rows, rank_metric=args.rank_metric)

    exported_segments = []
    if args.export_worst_k > 0:
        exported_segments = _export_worst_segments(
            run_dir=run_dir,
            worst_rows=ranked_rows[: args.export_worst_k],
            dataset_ctx=dataset_ctx,
            estimator=estimator,
            joint_adapter=joint_adapter,
            use_mask=args.use_mask,
            invalid_frame_mode=args.invalid_frame_mode,
            rank_metric=args.rank_metric,
        )

    elapsed_sec = time.time() - start_time
    summary = {
        "cfg": str(Path(args.cfg).expanduser().resolve()),
        "split": args.split,
        "camera": args.camera,
        "segment_length": int(args.segment_length),
        "segmentor_name": args.segmentor_name,
        "use_mask": bool(args.use_mask),
        "invalid_frame_mode": args.invalid_frame_mode,
        "rank_metric": args.rank_metric,
        "num_segments": len(segment_rows),
        "num_sequence_camera_pairs": len(grouped_rows),
        "num_exported_segments": len(exported_segments),
        "elapsed_sec": elapsed_sec,
        "overall_metrics": overall_metrics,
    }
    dump_json(run_dir / "run_summary.json", summary)
    print(
        "[sam3d-panoptic-segment-eval] completed "
        f"segments={len(segment_rows)} exported={len(exported_segments)} "
        f"elapsed={elapsed_sec:.1f}s run_dir={run_dir}"
    )


if __name__ == "__main__":
    main()
