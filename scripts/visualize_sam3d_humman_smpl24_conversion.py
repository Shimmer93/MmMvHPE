#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "sam-3d-body"))

from misc.official_mhr_smpl_conversion import (
    DEFAULT_MHR_ROOT,
    DEFAULT_SMPL_MODEL_PATH,
    OfficialSam3dToSmplConverter,
)
from misc.registry import create_dataset
from misc.sam3d_eval import sam3_cam_int_from_rgb_camera
from misc.skeleton import SMPLSkeleton
from misc.utils import load_cfg, merge_args_cfg
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info


class _MockArgs:
    checkpoint_path = ""
    gpus = 1
    num_workers = 0
    batch_size = 1
    batch_size_eva = 1
    pin_memory = False
    prefetch_factor = 2
    use_wandb = False
    save_test_preds = False


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_PANOPTIC_RUN_SEGMENT_EVAL = _load_module(
    "sam3d_panoptic_run_segment_eval_for_humman_vis",
    REPO_ROOT / "tools" / "sam3d_panoptic_segment_eval" / "run_segment_eval.py",
)
_load_estimator = _PANOPTIC_RUN_SEGMENT_EVAL._load_estimator
_process_image_for_display = _PANOPTIC_RUN_SEGMENT_EVAL._process_image_for_display
_world_to_camera = _PANOPTIC_RUN_SEGMENT_EVAL._world_to_camera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize HuMMan GT SMPL24 vs SAM MHR70 vs converted SAM SMPL24.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--cfg",
        default="configs/exp/humman/cross_camera_split/hpe.yml",
        help="HuMMan config path",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--camera", default="kinect_000", help="RGB camera to evaluate")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index after filtering to one camera")
    parser.add_argument("--checkpoint-root", default="/opt/data/SAM_3dbody_checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mhr-root", default=str(DEFAULT_MHR_ROOT))
    parser.add_argument("--smpl-model-path", default=str(DEFAULT_SMPL_MODEL_PATH))
    parser.add_argument("--segmentor-name", default="none", choices=["none", "sam2", "sam3"])
    parser.add_argument("--segmentor-path", default="/opt/data/SAM3_checkpoint")
    parser.add_argument("--use-mask", action="store_true")
    parser.add_argument(
        "--out-dir",
        default="logs/sam3d_humman_conversion_vis",
        help="Output directory root",
    )
    return parser.parse_args()


def _resolve_dataset_cfg(hparams: Any, split: str) -> tuple[dict, list]:
    if split == "train":
        return hparams.train_dataset, hparams.train_pipeline
    if split == "val":
        return hparams.val_dataset, hparams.val_pipeline
    if split == "test":
        return hparams.test_dataset, hparams.test_pipeline
    raise ValueError(f"Unsupported split={split}")


def _build_dataset(cfg_path: str, split: str, camera: str):
    cfg = load_cfg(cfg_path)
    hparams = merge_args_cfg(_MockArgs(), cfg)
    dataset_cfg, pipeline_cfg = _resolve_dataset_cfg(hparams, split)
    dataset_cfg = copy.deepcopy(dataset_cfg)
    pipeline_cfg = copy.deepcopy(pipeline_cfg)
    params = copy.deepcopy(dataset_cfg["params"])
    params["rgb_cameras"] = [camera]
    params["rgb_cameras_per_sample"] = 1
    if "depth_cameras" in params:
        params["depth_cameras"] = [camera]
        params["depth_cameras_per_sample"] = 1
    params["use_all_pairs"] = False
    dataset_cfg["params"] = params
    dataset, _ = create_dataset(dataset_cfg["name"], dataset_cfg["params"], pipeline_cfg)
    if len(dataset) == 0:
        raise ValueError(f"No samples found for camera {camera} in split {split}.")
    return hparams, dataset


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _extract_rgb_camera(sample: dict) -> dict:
    rgb_camera = sample["rgb_camera"]
    if isinstance(rgb_camera, list):
        if len(rgb_camera) != 1:
            raise ValueError(f"Expected one rgb_camera entry, got {len(rgb_camera)}.")
        rgb_camera = rgb_camera[0]
    return rgb_camera


def _mhr70_edges() -> list[tuple[int, int]]:
    keypoint_info = mhr70_pose_info["keypoint_info"]
    skeleton_info = mhr70_pose_info["skeleton_info"]
    name_to_id = {item["name"]: int(kpt_id) for kpt_id, item in keypoint_info.items()}
    edges = []
    for _, sk in skeleton_info.items():
        src_name, dst_name = sk["link"]
        if src_name in name_to_id and dst_name in name_to_id:
            edges.append((name_to_id[src_name], name_to_id[dst_name]))
    return edges


def _project_points(points_cam: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_cam, dtype=np.float32)
    K = np.asarray(intrinsic, dtype=np.float32)
    z = pts[:, 2:3]
    if np.any(z <= 1e-6):
        raise ValueError("Cannot project points with non-positive depth.")
    proj = (K @ pts.T).T
    return proj[:, :2] / proj[:, 2:3]


def _draw_skeleton(image_rgb: np.ndarray, points_2d: np.ndarray, bones: list[tuple[int, int]], color: tuple[int, int, int], title: str) -> np.ndarray:
    canvas = image_rgb.copy()
    h, w = canvas.shape[:2]
    pts = np.asarray(points_2d, dtype=np.float32)
    for i, j in bones:
        if i >= len(pts) or j >= len(pts):
            continue
        p1 = pts[i]
        p2 = pts[j]
        if not np.isfinite(p1).all() or not np.isfinite(p2).all():
            continue
        cv2.line(canvas, (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1]))), color, 2, cv2.LINE_AA)
    for p in pts:
        if not np.isfinite(p).all():
            continue
        if p[0] < -4 or p[0] > w + 4 or p[1] < -4 or p[1] > h + 4:
            continue
        cv2.circle(canvas, (int(round(p[0])), int(round(p[1]))), 3, color, -1, cv2.LINE_AA)
    cv2.putText(canvas, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return canvas


def _camera_to_plot(points_cam: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_cam, dtype=np.float32)
    return np.stack([pts[:, 0], pts[:, 2], -pts[:, 1]], axis=1)


def _plot_3d(ax, points_cam: np.ndarray, bones: list[tuple[int, int]], color: str, label: str) -> None:
    pts = _camera_to_plot(points_cam)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=12, c=color, label=label)
    for i, j in bones:
        if i >= len(pts) or j >= len(pts):
            continue
        seg = pts[[i, j]]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, linewidth=1.4, alpha=0.9)


def main() -> None:
    args = parse_args()
    hparams, dataset = _build_dataset(args.cfg, args.split, args.camera)
    sample_idx = max(0, min(args.sample_idx, len(dataset) - 1))
    sample = dataset[sample_idx]
    denorm_params = getattr(hparams, "vis_denorm_params", None)
    rgb_camera = _extract_rgb_camera(sample)
    rgb_image = _process_image_for_display(sample["input_rgb"], denorm_params, key="rgb")
    rgb_image = np.asarray(rgb_image, dtype=np.uint8)

    estimator = _load_estimator(
        Path(args.checkpoint_root).expanduser().resolve(),
        args.device,
        args.segmentor_name,
        args.segmentor_path,
    )
    outputs = estimator.process_one_image(
        rgb_image,
        cam_int=sam3_cam_int_from_rgb_camera(rgb_camera),
        use_mask=args.use_mask,
    )
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one SAM3D person output, got {len(outputs)}.")
    person = outputs[0]

    gt_keypoints = _to_numpy(sample["gt_keypoints"]).astype(np.float32)
    gt_camera = _world_to_camera(gt_keypoints, rgb_camera["extrinsic"])
    sam_mhr70_camera = np.asarray(person["pred_keypoints_3d"], dtype=np.float32) + np.asarray(
        person["pred_cam_t"], dtype=np.float32
    ).reshape(1, 3)
    converter = OfficialSam3dToSmplConverter(
        device=args.device,
        mhr_root=args.mhr_root,
        smpl_model_path=args.smpl_model_path,
        batch_size=1,
    )
    converted = converter.convert_outputs([person], batch_size=1, return_errors=True)
    sam_smpl24_camera = np.asarray(converted["smpl_joints24"][0], dtype=np.float32)

    intrinsic = np.asarray(rgb_camera["intrinsic"], dtype=np.float32)
    gt_2d = _project_points(gt_camera, intrinsic)
    sam_mhr70_2d = _project_points(sam_mhr70_camera, intrinsic)
    sam_smpl24_2d = _project_points(sam_smpl24_camera, intrinsic)

    gt_overlay = _draw_skeleton(rgb_image, gt_2d, SMPLSkeleton.bones, (0, 220, 80), "GT SMPL24")
    raw_overlay = _draw_skeleton(rgb_image, sam_mhr70_2d, _mhr70_edges(), (255, 140, 0), "SAM MHR70")
    converted_overlay = _draw_skeleton(rgb_image, sam_smpl24_2d, SMPLSkeleton.bones, (70, 140, 255), "Converted SAM SMPL24")

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        / Path(args.cfg).stem
        / args.split
        / args.camera
        / f"sample_{sample_idx:05d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "rgb_gt_smpl24_overlay.jpg"), cv2.cvtColor(gt_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "rgb_sam_mhr70_overlay.jpg"), cv2.cvtColor(raw_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "rgb_sam_smpl24_overlay.jpg"), cv2.cvtColor(converted_overlay, cv2.COLOR_RGB2BGR))

    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(gt_overlay)
    ax1.set_title("GT SMPL24 Overlay")
    ax1.axis("off")
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(raw_overlay)
    ax2.set_title("SAM MHR70 Overlay")
    ax2.axis("off")
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(converted_overlay)
    ax3.set_title("Converted SAM SMPL24 Overlay")
    ax3.axis("off")
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    _plot_3d(ax4, gt_camera, SMPLSkeleton.bones, "green", "GT SMPL24")
    _plot_3d(ax4, sam_mhr70_camera, _mhr70_edges(), "orange", "SAM MHR70")
    _plot_3d(ax4, sam_smpl24_camera, SMPLSkeleton.bones, "dodgerblue", "Converted SAM SMPL24")
    ax4.set_title("3D Camera-Space Skeletons")
    ax4.set_xlabel("x")
    ax4.set_ylabel("z")
    ax4.set_zlabel("-y")
    ax4.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_figure.png", dpi=160)
    plt.close(fig)

    np.savez_compressed(
        out_dir / "joint_sets.npz",
        gt_smpl24_camera=gt_camera,
        sam_mhr70_camera=sam_mhr70_camera,
        sam_smpl24_camera=sam_smpl24_camera,
        intrinsic=intrinsic,
    )
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "cfg": str(Path(args.cfg).expanduser().resolve()),
                "split": args.split,
                "camera": args.camera,
                "sample_idx": int(sample_idx),
                "sample_id": str(sample.get("sample_id", f"sample_{sample_idx}")),
                "segmentor_name": args.segmentor_name,
                "use_mask": bool(args.use_mask),
                "output_count": int(len(outputs)),
                "official_conversion_fitting_error": None
                if converted["fitting_errors"] is None
                else float(converted["fitting_errors"][0]),
            },
            f,
            indent=2,
            ensure_ascii=True,
        )
    print(f"[sam3d-humman-conversion] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
