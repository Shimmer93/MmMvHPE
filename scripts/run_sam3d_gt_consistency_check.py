#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.humman_dataset_v2 import HummanPreprocessedDatasetV2
from datasets.panoptic_preprocessed_dataset_v1 import PanopticPreprocessedDatasetV1
from metrics.mpjpe import mpjpe_func, pampjpe_func, pcmpjpe_func
from misc.official_mhr_smpl_conversion import (
    DEFAULT_MHR_ROOT,
    DEFAULT_SMPL_MODEL_PATH,
    OfficialSam3dToSmplConverter,
)
from misc.skeleton import PanopticCOCO19Skeleton, SMPLSkeleton
from projects.synthetic_data.export_formats import mhr70_to_panoptic19
from projects.synthetic_data.sam3d_adapter import SAM3DRunner
from projects.synthetic_data.visualization import _load_mhr70_edges


RGB_GT = (60, 210, 90)
RGB_PRED = (235, 80, 70)
RGB_MHR = (255, 160, 30)


@dataclass
class SampleEvalResult:
    dataset_name: str
    sample_id: str
    seq_name: str
    camera_name: str
    figure_path: Path
    mpjpe: float
    pa_mpjpe: float
    pc_mpjpe: float
    fitting_error: float | None
    status: str
    reason: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SAM-3D-Body predictions against HuMMan/Panoptic ground truth and save QC figures.",
        allow_abbrev=False,
    )
    parser.add_argument("--checkpoint-root", default="/opt/data/SAM_3dbody_checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="logs/sam3d_gt_consistency_check")
    parser.add_argument("--run-name", default="smoke")
    parser.add_argument("--humman-root", default="/opt/data/humman_cropped")
    parser.add_argument("--humman-max-samples", type=int, default=4)
    parser.add_argument("--humman-random-seed", type=int, default=0)
    parser.add_argument("--panoptic-root", default="/opt/data/panoptic_kinoptic_single_actor_cropped")
    parser.add_argument("--panoptic-sequences", default="171026_cello3")
    parser.add_argument("--panoptic-max-samples", type=int, default=4)
    parser.add_argument("--panoptic-random-seed", type=int, default=0)
    parser.add_argument("--use-mask", action="store_true", help="Enable SAM mask-conditioned inference if supported.")
    parser.add_argument("--mhr-root", default=str(DEFAULT_MHR_ROOT))
    parser.add_argument("--smpl-model-path", default=str(DEFAULT_SMPL_MODEL_PATH))
    parser.add_argument("--conversion-batch-size", type=int, default=32)
    return parser.parse_args()


def resolve_smpl_model_path(user_path: str) -> Path:
    candidates = [
        Path(user_path).expanduser(),
        REPO_ROOT / "weights" / "smpl" / "SMPL_NEUTRAL.pkl",
        REPO_ROOT / "weights" / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "Could not resolve an SMPL model path. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def ensure_single_rgb_frame(sample: dict[str, Any]) -> np.ndarray:
    frames = sample["input_rgb"]
    if isinstance(frames, list):
        if len(frames) != 1:
            raise ValueError(f"Expected exactly one RGB frame, got {len(frames)}.")
        frame = frames[0]
    else:
        frame = frames
    frame = np.asarray(frame, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected RGB frame shape (H,W,3), got {frame.shape}.")
    return frame


def points_world_to_camera(points_world: np.ndarray, extrinsic_world_to_camera: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_world, dtype=np.float32)
    ext = np.asarray(extrinsic_world_to_camera, dtype=np.float32).reshape(3, 4)
    r = ext[:, :3]
    t = ext[:, 3]
    return (pts @ r.T + t.reshape(1, 3)).astype(np.float32)


def points_camera_to_image(points_camera: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_camera, dtype=np.float32)
    k = np.asarray(intrinsic, dtype=np.float32).reshape(3, 3)
    z = np.clip(pts[:, 2], 1e-6, None)
    x = k[0, 0] * (pts[:, 0] / z) + k[0, 2]
    y = k[1, 1] * (pts[:, 1] / z) + k[1, 2]
    return np.stack([x, y], axis=1).astype(np.float32)


def _draw_bones_on_image(image_rgb: np.ndarray, points_2d: np.ndarray, bones: list[list[int]], color: tuple[int, int, int], *, thickness: int = 2, radius: int = 3) -> np.ndarray:
    canvas = np.asarray(image_rgb, dtype=np.uint8).copy()
    pts = np.asarray(points_2d, dtype=np.float32)
    h, w = canvas.shape[:2]
    for i, j in bones:
        if i >= pts.shape[0] or j >= pts.shape[0]:
            continue
        pi = pts[i]
        pj = pts[j]
        if not np.isfinite(pi).all() or not np.isfinite(pj).all():
            continue
        p1 = (int(round(float(pi[0]))), int(round(float(pi[1]))))
        p2 = (int(round(float(pj[0]))), int(round(float(pj[1]))))
        if (p1[0] < -w or p1[0] > 2 * w or p1[1] < -h or p1[1] > 2 * h or
                p2[0] < -w or p2[0] > 2 * w or p2[1] < -h or p2[1] > 2 * h):
            continue
        cv2.line(canvas, p1, p2, color, thickness, lineType=cv2.LINE_AA)
    for pt in pts:
        if not np.isfinite(pt).all():
            continue
        x = int(round(float(pt[0])))
        y = int(round(float(pt[1])))
        cv2.circle(canvas, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    return canvas


def draw_combined_target_overlay(image_rgb: np.ndarray, gt_2d: np.ndarray, pred_2d: np.ndarray, bones: list[list[int]]) -> np.ndarray:
    out = _draw_bones_on_image(image_rgb, gt_2d, bones, RGB_GT, thickness=2, radius=3)
    out = _draw_bones_on_image(out, pred_2d, bones, RGB_PRED, thickness=2, radius=3)
    return out


def mhr70_edges() -> list[list[int]]:
    return [[int(i), int(j)] for i, j, _ in _load_mhr70_edges()]


def set_axes_equal(ax, points: np.ndarray) -> None:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if pts.size == 0 or not np.isfinite(pts).all():
        return
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 1e-2)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_3d_skeleton(ax, points_3d: np.ndarray, bones: list[list[int]], *, color: tuple[float, float, float], title: str, alpha: float = 1.0) -> None:
    pts = np.asarray(points_3d, dtype=np.float32)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=8, color=color, alpha=alpha)
    for i, j in bones:
        if i >= pts.shape[0] or j >= pts.shape[0]:
            continue
        pi = pts[i]
        pj = pts[j]
        if not np.isfinite(pi).all() or not np.isfinite(pj).all():
            continue
        ax.plot(
            [pi[0], pj[0]],
            [pi[1], pj[1]],
            [pi[2], pj[2]],
            color=color,
            linewidth=1.0,
            alpha=alpha,
        )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=18, azim=35)
    set_axes_equal(ax, pts)


def plot_3d_overlay(ax, gt_points: np.ndarray, pred_points: np.ndarray, bones: list[list[int]], *, title: str) -> None:
    plot_3d_skeleton(ax, gt_points, bones, color=(0.2, 0.75, 0.35), title=title, alpha=0.9)
    plot_3d_skeleton(ax, pred_points, bones, color=(0.9, 0.2, 0.2), title=title, alpha=0.7)


def compute_metrics(pred: np.ndarray, gt: np.ndarray, *, pelvis_idx: int) -> tuple[float, float, float]:
    pred_batch = np.asarray(pred, dtype=np.float32)[None, ...]
    gt_batch = np.asarray(gt, dtype=np.float32)[None, ...]
    mpjpe = float(mpjpe_func(pred_batch, gt_batch, reduce=True))
    pa_mpjpe = float(pampjpe_func(pred_batch, gt_batch, reduce=True))
    pc_mpjpe = float(pcmpjpe_func(pred_batch, gt_batch, pelvis_idx=pelvis_idx, reduce=True))
    return mpjpe, pa_mpjpe, pc_mpjpe


def select_indices(num_items: int, max_samples: int) -> list[int]:
    if num_items <= 0 or max_samples <= 0:
        return []
    max_samples = min(num_items, max_samples)
    if max_samples == num_items:
        return list(range(num_items))
    idxs = np.linspace(0, num_items - 1, num=max_samples)
    out = sorted({int(round(float(i))) for i in idxs})
    while len(out) < max_samples:
        for i in range(num_items):
            if i not in out:
                out.append(i)
            if len(out) == max_samples:
                break
    return sorted(out[:max_samples])


def compute_visibility_stats(sample: dict[str, Any]) -> dict[str, float]:
    image_rgb = ensure_single_rgb_frame(sample)
    camera = sample["rgb_camera"]
    gt_camera = points_world_to_camera(sample["gt_keypoints"], camera["extrinsic"])
    gt_2d = points_camera_to_image(gt_camera, camera["intrinsic"])
    h, w = image_rgb.shape[:2]
    valid_depth = gt_camera[:, 2] > 0.0
    in_view = (
        valid_depth
        & (gt_2d[:, 0] >= 0.0)
        & (gt_2d[:, 0] < float(w))
        & (gt_2d[:, 1] >= 0.0)
        & (gt_2d[:, 1] < float(h))
    )
    if not np.any(in_view):
        return {
            "num_in_view": 0.0,
            "bbox_area": 0.0,
            "bbox_min_x": float("nan"),
            "bbox_max_x": float("nan"),
            "bbox_min_y": float("nan"),
            "bbox_max_y": float("nan"),
            "image_width": float(w),
            "image_height": float(h),
        }
    xs = gt_2d[in_view, 0]
    ys = gt_2d[in_view, 1]
    area = float(max(0.0, (xs.max() - xs.min()) * (ys.max() - ys.min())))
    return {
        "num_in_view": float(np.sum(in_view)),
        "bbox_area": area,
        "bbox_min_x": float(xs.min()),
        "bbox_max_x": float(xs.max()),
        "bbox_min_y": float(ys.min()),
        "bbox_max_y": float(ys.max()),
        "image_width": float(w),
        "image_height": float(h),
    }


def bbox_touches_border(visibility: dict[str, float], *, margin_px: float) -> bool:
    if visibility["num_in_view"] <= 0:
        return True
    return bool(
        visibility["bbox_min_x"] <= margin_px
        or visibility["bbox_min_y"] <= margin_px
        or visibility["bbox_max_x"] >= visibility["image_width"] - margin_px
        or visibility["bbox_max_y"] >= visibility["image_height"] - margin_px
    )


def candidate_indices_for_scan(total: int, max_samples: int, *, multiplier: int = 32) -> list[int]:
    return select_indices(total, max(1, max_samples) * multiplier)


def run_estimator(estimator, image_rgb: np.ndarray, intrinsic: np.ndarray, use_mask: bool) -> tuple[list[dict[str, Any]] | None, str | None]:
    cam_int = torch.from_numpy(np.asarray(intrinsic, dtype=np.float32)).unsqueeze(0)
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            outputs = estimator.process_one_image(
                image_rgb,
                cam_int=cam_int,
                use_mask=use_mask,
            )
        return outputs, None
    except Exception as exc:  # noqa: BLE001
        captured = (stdout_buffer.getvalue() + "\n" + stderr_buffer.getvalue()).strip()
        detail = f"{type(exc).__name__}: {exc}"
        if captured:
            detail = f"{detail} | sam3d_log={captured.splitlines()[-1]}"
        return None, detail


def render_result_figure(
    path: Path,
    *,
    image_rgb: np.ndarray,
    gt_target_camera: np.ndarray,
    pred_target_camera: np.ndarray,
    pred_mhr_camera: np.ndarray,
    intrinsic: np.ndarray,
    target_bones: list[list[int]],
    target_title: str,
    text_lines: list[str],
) -> None:
    gt_target_2d = points_camera_to_image(gt_target_camera, intrinsic)
    pred_target_2d = points_camera_to_image(pred_target_camera, intrinsic)
    pred_mhr_2d = points_camera_to_image(pred_mhr_camera, intrinsic)

    overlay_target = draw_combined_target_overlay(image_rgb, gt_target_2d, pred_target_2d, target_bones)
    overlay_mhr = _draw_bones_on_image(image_rgb, pred_mhr_2d, mhr70_edges(), RGB_MHR, thickness=1, radius=2)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 10))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(image_rgb)
    ax1.set_title("Source RGB")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(overlay_target)
    ax2.set_title(f"{target_title} 2D Overlay")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(overlay_mhr)
    ax3.set_title("Raw MHR70 2D Projection")
    ax3.axis("off")

    ax4 = fig.add_subplot(2, 3, 4, projection="3d")
    plot_3d_overlay(ax4, gt_target_camera, pred_target_camera, target_bones, title=f"{target_title} Camera 3D")

    ax5 = fig.add_subplot(2, 3, 5, projection="3d")
    plot_3d_skeleton(ax5, pred_mhr_camera, mhr70_edges(), color=(1.0, 0.6, 0.0), title="Raw MHR70 Camera 3D")

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    ax6.text(
        0.0,
        1.0,
        "\n".join(text_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_contact_sheet(image_paths: list[Path], out_path: Path, *, title: str) -> None:
    if not image_paths:
        return
    cols = 2
    rows = int(math.ceil(len(image_paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 5))
    axes_arr = np.array(axes).reshape(rows, cols)
    for ax in axes_arr.reshape(-1):
        ax.axis("off")
    for ax, image_path in zip(axes_arr.reshape(-1), image_paths):
        img = plt.imread(image_path)
        ax.imshow(img)
        ax.set_title(image_path.stem)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def summarize_results(results: list[SampleEvalResult]) -> dict[str, Any]:
    valid = [r for r in results if r.status == "ok"]
    return {
        "num_total": int(len(results)),
        "num_valid": int(len(valid)),
        "num_invalid": int(len(results) - len(valid)),
        "mpjpe_mean": float(np.mean([r.mpjpe for r in valid])) if valid else float("nan"),
        "pa_mpjpe_mean": float(np.mean([r.pa_mpjpe for r in valid])) if valid else float("nan"),
        "pc_mpjpe_mean": float(np.mean([r.pc_mpjpe for r in valid])) if valid else float("nan"),
        "results": [
            {
                "dataset_name": r.dataset_name,
                "sample_id": r.sample_id,
                "seq_name": r.seq_name,
                "camera_name": r.camera_name,
                "figure_path": str(r.figure_path),
                "mpjpe": float(r.mpjpe),
                "pa_mpjpe": float(r.pa_mpjpe),
                "pc_mpjpe": float(r.pc_mpjpe),
                "fitting_error": None if r.fitting_error is None else float(r.fitting_error),
                "status": r.status,
                "reason": r.reason,
            }
            for r in results
        ],
    }


def evaluate_humman(
    *,
    output_dir: Path,
    dataset: HummanPreprocessedDatasetV2,
    max_samples: int,
    runner: SAM3DRunner,
    converter: OfficialSam3dToSmplConverter,
    use_mask: bool,
) -> list[SampleEvalResult]:
    results: list[SampleEvalResult] = []
    figure_paths: list[Path] = []
    bones = [list(x) for x in SMPLSkeleton.bones]
    accepted = 0
    for sample_index in candidate_indices_for_scan(len(dataset), max_samples):
        sample = dataset[sample_index]
        visibility = compute_visibility_stats(sample)
        if (
            visibility["num_in_view"] < 12
            or visibility["bbox_area"] < 3000.0
            or bbox_touches_border(visibility, margin_px=8.0)
        ):
            continue
        image_rgb = ensure_single_rgb_frame(sample)
        camera = sample["rgb_camera"]
        outputs, error_reason = run_estimator(runner.estimator, image_rgb, camera["intrinsic"], use_mask)
        fig_path = output_dir / "figures" / f"{sample['sample_id']}.png"
        if outputs is None or len(outputs) != 1:
            results.append(
                SampleEvalResult(
                    dataset_name="humman",
                    sample_id=str(sample["sample_id"]),
                    seq_name=str(sample["seq_name"]),
                    camera_name=str(sample["selected_cameras"]["rgb"][0]),
                    figure_path=fig_path,
                    mpjpe=float("nan"),
                    pa_mpjpe=float("nan"),
                    pc_mpjpe=float("nan"),
                    fitting_error=None,
                    status="invalid_prediction",
                    reason=error_reason or f"expected_single_person_got_{0 if outputs is None else len(outputs)}",
                )
            )
            continue

        output = outputs[0]
        pred_mhr_camera = (
            np.asarray(output["pred_keypoints_3d"], dtype=np.float32)
            + np.asarray(output["pred_cam_t"], dtype=np.float32).reshape(1, 3)
        )
        converted = converter.convert_outputs([output], return_vertices=False, return_errors=True)
        pred_target_camera = np.asarray(converted["smpl_joints24"], dtype=np.float32)[0]
        gt_target_camera = points_world_to_camera(sample["gt_keypoints"], camera["extrinsic"])
        mpjpe, pa_mpjpe, pc_mpjpe = compute_metrics(pred_target_camera, gt_target_camera, pelvis_idx=0)
        fitting_errors = converted.get("fitting_errors")
        fitting_error = None if fitting_errors is None else float(np.asarray(fitting_errors, dtype=np.float32)[0])
        render_result_figure(
            fig_path,
            image_rgb=image_rgb,
            gt_target_camera=gt_target_camera,
            pred_target_camera=pred_target_camera,
            pred_mhr_camera=pred_mhr_camera,
            intrinsic=camera["intrinsic"],
            target_bones=bones,
            target_title="HuMMan SMPL24",
            text_lines=[
                f"dataset      : humman",
                f"sample_id    : {sample['sample_id']}",
                f"seq_name     : {sample['seq_name']}",
                f"camera       : {sample['selected_cameras']['rgb'][0]}",
                f"mpjpe        : {mpjpe:.4f}",
                f"pa_mpjpe     : {pa_mpjpe:.4f}",
                f"pc_mpjpe     : {pc_mpjpe:.4f}",
                f"fit_error    : {fitting_error if fitting_error is not None else float('nan'):.6f}",
                f"sample_index : {sample_index}",
            ],
        )
        figure_paths.append(fig_path)
        results.append(
            SampleEvalResult(
                dataset_name="humman",
                sample_id=str(sample["sample_id"]),
                seq_name=str(sample["seq_name"]),
                camera_name=str(sample["selected_cameras"]["rgb"][0]),
                figure_path=fig_path,
                mpjpe=mpjpe,
                pa_mpjpe=pa_mpjpe,
                pc_mpjpe=pc_mpjpe,
                fitting_error=fitting_error,
                status="ok",
                reason=None,
            )
        )
        accepted += 1
        if accepted >= max_samples:
            break
    save_contact_sheet(figure_paths, output_dir / "index.png", title="HuMMan SAM3D Consistency Check")
    return results


def evaluate_panoptic(
    *,
    output_dir: Path,
    dataset: PanopticPreprocessedDatasetV1,
    max_samples: int,
    runner: SAM3DRunner,
    use_mask: bool,
) -> list[SampleEvalResult]:
    results: list[SampleEvalResult] = []
    figure_paths: list[Path] = []
    bones = [list(x) for x in PanopticCOCO19Skeleton.bones]
    accepted = 0
    for sample_index in candidate_indices_for_scan(len(dataset), max_samples):
        sample = dataset[sample_index]
        visibility = compute_visibility_stats(sample)
        if (
            visibility["num_in_view"] < 12
            or visibility["bbox_area"] < 2500.0
            or bbox_touches_border(visibility, margin_px=8.0)
        ):
            continue
        image_rgb = ensure_single_rgb_frame(sample)
        camera = sample["rgb_camera"]
        outputs, error_reason = run_estimator(runner.estimator, image_rgb, camera["intrinsic"], use_mask)
        fig_path = output_dir / "figures" / f"{sample['sample_id']}.png"
        if outputs is None or len(outputs) != 1:
            results.append(
                SampleEvalResult(
                    dataset_name="panoptic",
                    sample_id=str(sample["sample_id"]),
                    seq_name=str(sample["seq_name"]),
                    camera_name=str(sample["selected_cameras"]["rgb"][0]),
                    figure_path=fig_path,
                    mpjpe=float("nan"),
                    pa_mpjpe=float("nan"),
                    pc_mpjpe=float("nan"),
                    fitting_error=None,
                    status="invalid_prediction",
                    reason=error_reason or f"expected_single_person_got_{0 if outputs is None else len(outputs)}",
                )
            )
            continue

        output = outputs[0]
        pred_mhr_camera = (
            np.asarray(output["pred_keypoints_3d"], dtype=np.float32)
            + np.asarray(output["pred_cam_t"], dtype=np.float32).reshape(1, 3)
        )
        pred_target_camera = mhr70_to_panoptic19(pred_mhr_camera)
        gt_target_camera = points_world_to_camera(sample["gt_keypoints"], camera["extrinsic"])
        mpjpe, pa_mpjpe, pc_mpjpe = compute_metrics(pred_target_camera, gt_target_camera, pelvis_idx=2)
        render_result_figure(
            fig_path,
            image_rgb=image_rgb,
            gt_target_camera=gt_target_camera,
            pred_target_camera=pred_target_camera,
            pred_mhr_camera=pred_mhr_camera,
            intrinsic=camera["intrinsic"],
            target_bones=bones,
            target_title="Panoptic19",
            text_lines=[
                f"dataset      : panoptic",
                f"sample_id    : {sample['sample_id']}",
                f"seq_name     : {sample['seq_name']}",
                f"camera       : {sample['selected_cameras']['rgb'][0]}",
                f"mpjpe        : {mpjpe:.4f}",
                f"pa_mpjpe     : {pa_mpjpe:.4f}",
                f"pc_mpjpe     : {pc_mpjpe:.4f}",
                f"sample_index : {sample_index}",
            ],
        )
        figure_paths.append(fig_path)
        results.append(
            SampleEvalResult(
                dataset_name="panoptic",
                sample_id=str(sample["sample_id"]),
                seq_name=str(sample["seq_name"]),
                camera_name=str(sample["selected_cameras"]["rgb"][0]),
                figure_path=fig_path,
                mpjpe=mpjpe,
                pa_mpjpe=pa_mpjpe,
                pc_mpjpe=pc_mpjpe,
                fitting_error=None,
                status="ok",
                reason=None,
            )
        )
        accepted += 1
        if accepted >= max_samples:
            break
    save_contact_sheet(figure_paths, output_dir / "index.png", title="Panoptic SAM3D Consistency Check")
    return results


def build_humman_dataset(root: str, seed: int) -> HummanPreprocessedDatasetV2:
    random.seed(seed)
    return HummanPreprocessedDatasetV2(
        data_root=root,
        split="test",
        split_config="configs/datasets/humman_split_config.yml",
        split_to_use="cross_camera_split",
        test_mode=True,
        modality_names=["rgb"],
        seq_len=1,
        seq_step=1,
        use_all_pairs=False,
        skeleton_only=True,
        apply_to_new_world=True,
        remove_root_rotation=True,
    )


def build_panoptic_dataset(root: str, sequences: list[str], seed: int) -> PanopticPreprocessedDatasetV1:
    return PanopticPreprocessedDatasetV1(
        data_root=root,
        split="test",
        split_config=None,
        test_mode=True,
        modality_names=["rgb"],
        seq_len=1,
        seq_step=1,
        skeleton_only=True,
        apply_to_new_world=False,
        remove_root_rotation=False,
        random_seed=seed,
        sequence_allowlist=sequences,
    )


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve() / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)

    runner = SAM3DRunner(REPO_ROOT, args.checkpoint_root, device=args.device)
    converter = OfficialSam3dToSmplConverter(
        device=args.device,
        mhr_root=args.mhr_root,
        smpl_model_path=resolve_smpl_model_path(args.smpl_model_path),
        batch_size=args.conversion_batch_size,
    )

    humman_dataset = build_humman_dataset(args.humman_root, args.humman_random_seed)
    panoptic_sequences = [x.strip() for x in str(args.panoptic_sequences).split(",") if x.strip()]
    panoptic_dataset = build_panoptic_dataset(args.panoptic_root, panoptic_sequences, args.panoptic_random_seed)

    humman_results = evaluate_humman(
        output_dir=output_root / "humman",
        dataset=humman_dataset,
        max_samples=args.humman_max_samples,
        runner=runner,
        converter=converter,
        use_mask=bool(args.use_mask),
    )
    panoptic_results = evaluate_panoptic(
        output_dir=output_root / "panoptic",
        dataset=panoptic_dataset,
        max_samples=args.panoptic_max_samples,
        runner=runner,
        use_mask=bool(args.use_mask),
    )

    summary = {
        "run_dir": str(output_root),
        "humman": summarize_results(humman_results),
        "panoptic": summarize_results(panoptic_results),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
