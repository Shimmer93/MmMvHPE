#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

# Ensure this repository is first on import path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for p in list(sys.path):
    if p.endswith("/models/pc_encoders/modules"):
        sys.path.remove(p)

from misc.registry import create_dataset
from misc.skeleton import PanopticCOCO19Skeleton


FRAME_RE = re.compile(r"_(\d{8})$")


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _extract_last_rgb(input_rgb: Any, rgb_mean: list[float], rgb_std: list[float]) -> np.ndarray:
    arr = _to_numpy(input_rgb)
    # Input from ToTensor likely: T,C,H,W or V,T,C,H,W
    if arr.ndim == 5:
        arr = arr[0, -1]  # first view, last frame: C,H,W
    elif arr.ndim == 4:
        arr = arr[-1]  # last frame: C,H,W
    elif arr.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected input_rgb shape: {arr.shape}")

    if arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    mean = np.asarray(rgb_mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(rgb_std, dtype=np.float32).reshape(1, 1, 3)
    img = arr.astype(np.float32) * std + mean
    img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    return img


def _project_points(kpts_world: np.ndarray, K: np.ndarray, ext: np.ndarray) -> np.ndarray:
    R = ext[:, :3]
    T = ext[:, 3:4]
    cam = (R @ kpts_world.T + T).T
    z = cam[:, 2:3]
    z_safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
    uvw = (K @ cam.T).T
    uv = uvw[:, :2] / z_safe
    return uv.astype(np.float32)


def _draw_skeleton_on_image(img: np.ndarray, uv: np.ndarray, bones: list[list[int]]) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for i, j in bones:
        if i >= uv.shape[0] or j >= uv.shape[0]:
            continue
        p1 = uv[i]
        p2 = uv[j]
        if (
            np.isfinite(p1).all()
            and np.isfinite(p2).all()
            and -1000 <= p1[0] <= w + 1000
            and -1000 <= p1[1] <= h + 1000
            and -1000 <= p2[0] <= w + 1000
            and -1000 <= p2[1] <= h + 1000
        ):
            cv2.line(out, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2, cv2.LINE_AA)
    for p in uv:
        if np.isfinite(p).all():
            cv2.circle(out, (int(p[0]), int(p[1])), 3, (255, 50, 50), -1, cv2.LINE_AA)
    return out


def _render_plane(
    kpts: np.ndarray,
    bones: list[list[int]],
    dims: tuple[int, int],
    limits: tuple[float, float, float, float],
    size: tuple[int, int],
    title: str,
) -> np.ndarray:
    w, h = size
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    x_min, x_max, y_min, y_max = limits
    margin = 24
    cv2.rectangle(canvas, (margin, margin), (w - margin, h - margin), (220, 220, 220), 1)
    cv2.putText(canvas, title, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)

    def map_pt(pt2: np.ndarray) -> tuple[int, int]:
        x = float(pt2[0])
        y = float(pt2[1])
        tx = 0.0 if x_max <= x_min else (x - x_min) / (x_max - x_min)
        ty = 0.0 if y_max <= y_min else (y - y_min) / (y_max - y_min)
        px = int(margin + tx * (w - 2 * margin))
        py = int(h - margin - ty * (h - 2 * margin))
        return px, py

    pts2 = kpts[:, list(dims)]
    for i, j in bones:
        if i >= pts2.shape[0] or j >= pts2.shape[0]:
            continue
        p1 = map_pt(pts2[i])
        p2 = map_pt(pts2[j])
        cv2.line(canvas, p1, p2, (0, 130, 0), 2, cv2.LINE_AA)

    for p in pts2:
        cv2.circle(canvas, map_pt(p), 3, (30, 30, 220), -1, cv2.LINE_AA)
    return canvas


def _render_metrics_panel(
    bodycenter_hist: list[float],
    hip_err_hist: list[float],
    up_err_hist: list[float],
    size: tuple[int, int],
) -> np.ndarray:
    w, h = size
    c = np.full((h, w, 3), 255, dtype=np.uint8)
    margin = 26
    cv2.putText(c, "Stability Metrics", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(c, "blue=BodyCenter norm, green=hip->+X err, red=up->+Y err", (16, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.rectangle(c, (margin, 60), (w - margin, h - margin), (220, 220, 220), 1)

    n = max(len(bodycenter_hist), 1)
    x0, y0 = margin, 60
    x1, y1 = w - margin, h - margin
    plot_w = x1 - x0
    plot_h = y1 - y0

    ymax = max(
        1.0,
        float(max(bodycenter_hist) if bodycenter_hist else 0.0) * 1.2,
        float(max(hip_err_hist) if hip_err_hist else 0.0) * 1.2,
        float(max(up_err_hist) if up_err_hist else 0.0) * 1.2,
    )

    def draw_series(vals: list[float], color: tuple[int, int, int]) -> None:
        if not vals:
            return
        pts = []
        for i, v in enumerate(vals):
            tx = 0.0 if n <= 1 else i / (n - 1)
            ty = min(max(float(v) / ymax, 0.0), 1.0)
            px = int(x0 + tx * plot_w)
            py = int(y1 - ty * plot_h)
            pts.append((px, py))
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(c, a, b, color, 2, cv2.LINE_AA)

    draw_series(bodycenter_hist, (210, 70, 40))
    draw_series(hip_err_hist, (20, 170, 20))
    draw_series(up_err_hist, (40, 40, 220))
    return c


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a short MP4 to inspect pelvis-centering/root-rotation stability on Panoptic."
    )
    p.add_argument("--config", type=Path, default=Path("configs/exp/panoptic/cross_camera_split/hpe.yml"))
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--sequence", type=str, required=True)
    p.add_argument("--camera", type=str, default="kinect_001")
    p.add_argument("--max-frames", type=int, default=150)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument(
        "--out-video",
        type=Path,
        default=Path("logs/panoptic_stability/panoptic_keypoints_stability.mp4"),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_key = f"{args.split}_dataset"
    pipeline_key = f"{args.split}_pipeline"
    if dataset_key not in cfg or pipeline_key not in cfg:
        raise KeyError(f"Config must contain `{dataset_key}` and `{pipeline_key}`.")

    dataset_cfg = cfg[dataset_key]
    dataset_params = dict(dataset_cfg.get("params", {}))
    dataset_params["sequence_allowlist"] = [args.sequence]
    dataset_params["seq_len"] = 1
    dataset_params["use_all_pairs"] = True
    dataset_params["rgb_cameras"] = [args.camera]
    dataset_params["depth_cameras"] = [args.camera]
    dataset_params["strict_validation"] = True

    dataset, _ = create_dataset(dataset_cfg["name"], dataset_params, cfg[pipeline_key])
    if len(dataset) == 0:
        raise ValueError("Dataset has zero samples with the selected split/sequence/camera.")

    # Select indices for requested sequence/camera, ordered by frame id.
    idx_and_frame: list[tuple[int, int]] = []
    for idx, info in enumerate(dataset.data_list):
        if info.get("seq_name") != args.sequence:
            continue
        if info.get("rgb_camera") != args.camera:
            continue
        frame_ids = dataset.sequence_data[args.sequence]["frame_ids"]
        start = int(info["start_frame"])
        if start < 0 or start >= len(frame_ids):
            continue
        idx_and_frame.append((idx, int(frame_ids[start])))

    idx_and_frame.sort(key=lambda x: x[1])
    if not idx_and_frame:
        raise ValueError(
            f"No matching samples for sequence={args.sequence}, camera={args.camera}."
        )
    idx_and_frame = idx_and_frame[: args.max_frames]

    denorm = cfg.get("vis_denorm_params", {})
    rgb_mean = denorm.get("rgb_mean", [123.675, 116.28, 103.53])
    rgb_std = denorm.get("rgb_std", [58.395, 57.12, 57.375])

    frames_data = []
    xy_all = []
    xz_all = []
    bodycenter_hist: list[float] = []
    hip_err_hist: list[float] = []
    up_err_hist: list[float] = []

    for idx, fid in idx_and_frame:
        s = dataset[idx]
        kpts = _to_numpy(s["gt_keypoints"]).astype(np.float32)
        if kpts.shape[0] <= 12:
            raise ValueError(f"Expected Panoptic 19 joints, got {kpts.shape}")
        rgb = _extract_last_rgb(s["input_rgb"], rgb_mean, rgb_std)
        cam = s["rgb_camera"]
        K = _to_numpy(cam["intrinsic"]).astype(np.float32)
        ext = _to_numpy(cam["extrinsic"]).astype(np.float32)
        uv = _project_points(kpts, K, ext)
        overlay = _draw_skeleton_on_image(rgb, uv, PanopticCOCO19Skeleton.bones)

        bodycenter = kpts[2]
        neck = kpts[0]
        lhip = kpts[12]
        rhip = kpts[9]
        hip_vec = rhip - lhip
        up_vec = neck - bodycenter
        hip_norm = float(np.linalg.norm(hip_vec)) + 1e-9
        up_norm = float(np.linalg.norm(up_vec)) + 1e-9
        hip_unit = hip_vec / hip_norm
        up_unit = up_vec / up_norm
        hip_err = math.degrees(math.acos(float(np.clip(np.dot(hip_unit, np.array([1.0, 0.0, 0.0], dtype=np.float32)), -1.0, 1.0))))
        up_err = math.degrees(math.acos(float(np.clip(np.dot(up_unit, np.array([0.0, 1.0, 0.0], dtype=np.float32)), -1.0, 1.0))))

        frames_data.append(
            {
                "frame_id": fid,
                "overlay": overlay,
                "kpts": kpts,
                "bodycenter_norm": float(np.linalg.norm(bodycenter)),
                "hip_err": float(hip_err),
                "up_err": float(up_err),
            }
        )
        xy_all.append(kpts[:, [0, 1]])
        xz_all.append(kpts[:, [0, 2]])
        bodycenter_hist.append(float(np.linalg.norm(bodycenter)))
        hip_err_hist.append(float(hip_err))
        up_err_hist.append(float(up_err))

    xy_cat = np.concatenate(xy_all, axis=0)
    xz_cat = np.concatenate(xz_all, axis=0)
    xy_limits = (
        float(xy_cat[:, 0].min() - 0.05),
        float(xy_cat[:, 0].max() + 0.05),
        float(xy_cat[:, 1].min() - 0.05),
        float(xy_cat[:, 1].max() + 0.05),
    )
    xz_limits = (
        float(xz_cat[:, 0].min() - 0.05),
        float(xz_cat[:, 0].max() + 0.05),
        float(xz_cat[:, 1].min() - 0.05),
        float(xz_cat[:, 1].max() + 0.05),
    )

    panel_size = (640, 360)
    out_h, out_w = panel_size[1] * 2, panel_size[0] * 2
    args.out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (out_w, out_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {args.out_video}")

    for t, fd in enumerate(frames_data):
        p0 = cv2.resize(fd["overlay"], panel_size, interpolation=cv2.INTER_LINEAR)
        cv2.putText(
            p0,
            f"RGB Overlay  seq={args.sequence} cam={args.camera} frame={fd['frame_id']}",
            (10, panel_size[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        p1 = _render_plane(fd["kpts"], PanopticCOCO19Skeleton.bones, (0, 1), xy_limits, panel_size, "XY (new_world)")
        p2 = _render_plane(fd["kpts"], PanopticCOCO19Skeleton.bones, (0, 2), xz_limits, panel_size, "XZ (new_world)")
        p3 = _render_metrics_panel(bodycenter_hist[: t + 1], hip_err_hist[: t + 1], up_err_hist[: t + 1], panel_size)
        cv2.putText(
            p3,
            f"BodyCenter norm={fd['bodycenter_norm']:.5f}  hip_err={fd['hip_err']:.2f}deg  up_err={fd['up_err']:.2f}deg",
            (12, panel_size[1] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (10, 10, 10),
            1,
            cv2.LINE_AA,
        )

        top = np.concatenate([p0, p1], axis=1)
        bot = np.concatenate([p2, p3], axis=1)
        frame = np.concatenate([top, bot], axis=0)
        writer.write(frame)

    writer.release()
    print(f"[panoptic-stability-video] wrote: {args.out_video}")
    print(f"[panoptic-stability-video] frames: {len(frames_data)} fps: {args.fps}")


if __name__ == "__main__":
    main()
