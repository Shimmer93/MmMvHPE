"""
Visualize prediction PKL files and map keypoints/meshes to global/world coordinates
using HUMMAN raw camera extrinsics from `<data_root>/cameras`.

Example:
    uv run python tools/vis_and_save_global.py \
      --pred-file /root/MmMvHPE/logs/HummanLEIR_test_predictions.pkl \
      --data-root /root/autodl-tmp/humman_cropped \
      --save-dir /root/autodl-tmp/logs/vis_global \
      --start 0 --end 99 \
      --sensor-modality lidar \
      --sensor-idx 0 \
      --lidar-frame-mode seq_fixed
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from misc.utils import load
from models.smpl import SMPL
from tools.vis_and_save import (
    _clean_axis_2d,
    _clean_axis_3d,
    _draw_skeleton_2d,
    _draw_skeleton_3d,
    _extract_sample_any,
    _get_pelvis_from_keypoints,
    _get_pelvis_from_smpl_joints,
    _get_sample_keypoints,
    _infer_skl_format,
    _load_humman_inputs,
    _load_smpl_vertices,
    _plot_mesh_2d_dims,
    _plot_mesh_3d,
    _set_2d_limits,
    _set_3d_limits_data_aspect,
    _to_numpy_float,
    _to_save_name,
    _transform_points,
)

_HUMMAN_ID_RE = re.compile(
    r"^(?P<seq>p\d+_a\d+)_rgb_(?P<rgb>kinect_\d{3}|iphone)_depth_"
    r"(?P<depth>kinect_\d{3}|iphone)_(?P<frame>\d+)$"
)


def _parse_humman_sample_id(sample_id: str) -> Tuple[str, str, str]:
    match = _HUMMAN_ID_RE.match(str(sample_id))
    if match is None:
        raise ValueError(f"Unexpected HUMMAN sample_id format: {sample_id}")
    return match.group("seq"), match.group("rgb"), match.group("depth")


def _camera_name_to_key(camera_name: str, modality: str) -> str:
    if camera_name.startswith("kinect"):
        suffix = camera_name.split("_")[1]
        if modality == "rgb":
            return f"kinect_color_{suffix}"
        if modality in {"depth", "lidar"}:
            return f"kinect_depth_{suffix}"
    return "iphone"


def _load_world_extrinsic_from_humman_camera(
    sample_id: str,
    data_root: str,
    modality: str,
    camera_cache: Dict[str, Dict[str, dict]],
) -> np.ndarray:
    seq_name, rgb_cam, depth_cam = _parse_humman_sample_id(sample_id)
    mod = str(modality).lower()
    if mod == "rgb":
        camera_name = rgb_cam
    elif mod in {"depth", "lidar"}:
        camera_name = depth_cam
    else:
        raise ValueError(f"Unsupported modality `{modality}` for HUMMAN camera mapping.")

    if seq_name not in camera_cache:
        camera_file = osp.join(data_root, "cameras", f"{seq_name}_cameras.json")
        with open(camera_file, "r", encoding="utf-8") as f:
            camera_cache[seq_name] = json.load(f)
    cameras = camera_cache[seq_name]

    cam_key = _camera_name_to_key(camera_name, mod)
    if cam_key not in cameras:
        raise KeyError(f"Camera key `{cam_key}` not found for sample `{sample_id}`.")

    cam = cameras[cam_key]
    r = np.asarray(cam["R"], dtype=np.float32)
    t = np.asarray(cam["T"], dtype=np.float32).reshape(3)
    if r.shape != (3, 3):
        raise ValueError(f"Invalid camera rotation shape for `{cam_key}`: {r.shape}")
    if not np.isfinite(r).all() or not np.isfinite(t).all():
        raise ValueError(f"Camera extrinsic for `{cam_key}` contains non-finite values.")

    # HUMMAN stores world->camera as X_cam = R * X_world + T (column-vector form).
    # Converted for _transform_points row-vector form: X_world = X_cam @ R + (-T @ R).
    world_r = r.T
    world_t = (-t.reshape(1, 3) @ r).reshape(3)
    return np.concatenate([world_r, world_t.reshape(3, 1)], axis=1).astype(np.float32)


def _valid_j3(points: Optional[np.ndarray]) -> bool:
    if points is None:
        return False
    arr = np.asarray(points)
    return arr.ndim == 2 and arr.shape[1] == 3


def _align_mesh_to_keypoints(
    mesh_vertices: Optional[np.ndarray],
    mesh_joints: Optional[np.ndarray],
    target_keypoints: Optional[np.ndarray],
    skl_format: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if mesh_vertices is None or mesh_joints is None or target_keypoints is None:
        return mesh_vertices, mesh_joints
    pelvis_target = _get_pelvis_from_keypoints(target_keypoints, skl_format)
    pelvis_mesh = _get_pelvis_from_smpl_joints(mesh_joints, skl_format)
    if pelvis_target is None or pelvis_mesh is None:
        return mesh_vertices, mesh_joints
    offset = pelvis_target.reshape(1, 3) - pelvis_mesh.reshape(1, 3)
    return mesh_vertices + offset, mesh_joints + offset


def _save_world_3d(
    out_path: str,
    mode: str,
    pc_world: np.ndarray,
    pred_kps_world: np.ndarray,
    gt_kps_world: Optional[np.ndarray],
    mesh_vertices: Optional[np.ndarray],
    gt_mesh_vertices: Optional[np.ndarray],
    faces: np.ndarray,
    pred_bone_color: str,
    pred_joint_color: str,
    gt_bone_color: str,
    gt_joint_color: str,
    pred_mesh_color: str,
    gt_mesh_color: str,
):
    fig = plt.figure(figsize=(6.2, 6.0), dpi=130)
    ax = fig.add_subplot(111, projection="3d")
    points_for_limits = []
    if pc_world.size > 0:
        ax.scatter(pc_world[:, 0], pc_world[:, 1], pc_world[:, 2], s=0.3, c="gray", alpha=0.20)
        points_for_limits.append(pc_world)
    if mode in {"pred", "overlay"}:
        _plot_mesh_3d(ax, mesh_vertices, faces, pred_mesh_color, alpha=0.32)
        _draw_skeleton_3d(
            ax,
            pred_kps_world,
            bone_color=pred_bone_color,
            joint_color=pred_joint_color,
            linewidth=1.7,
            joint_size=28.0,
        )
        points_for_limits.append(pred_kps_world)
        if mesh_vertices is not None:
            points_for_limits.append(mesh_vertices)
    if mode in {"gt", "overlay"} and _valid_j3(gt_kps_world):
        _plot_mesh_3d(ax, gt_mesh_vertices, faces, gt_mesh_color, alpha=0.28)
        _draw_skeleton_3d(
            ax,
            gt_kps_world,
            bone_color=gt_bone_color,
            joint_color=gt_joint_color,
            linewidth=1.6,
            joint_size=26.0,
        )
        points_for_limits.append(gt_kps_world)
        if gt_mesh_vertices is not None:
            points_for_limits.append(gt_mesh_vertices)
    _set_3d_limits_data_aspect(ax, points_for_limits, pad_ratio=0.03)
    _clean_axis_3d(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _save_world_2d_view(
    out_path: str,
    mode: str,
    dims: Tuple[int, int],
    invert_y: bool,
    pred_kps_world: np.ndarray,
    gt_kps_world: Optional[np.ndarray],
    mesh_vertices: Optional[np.ndarray],
    gt_mesh_vertices: Optional[np.ndarray],
    faces: np.ndarray,
    pred_bone_color: str,
    pred_joint_color: str,
    gt_bone_color: str,
    gt_joint_color: str,
    pred_mesh_color: str,
    gt_mesh_color: str,
):
    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=130)
    points_for_limit = []
    if mode in {"pred", "overlay"}:
        _plot_mesh_2d_dims(ax, mesh_vertices, faces, dims=dims, color=pred_mesh_color, alpha=0.30)
        pred_2d = pred_kps_world[:, list(dims)]
        _draw_skeleton_2d(
            ax,
            pred_2d,
            bone_color=pred_bone_color,
            joint_color=pred_joint_color,
            linewidth=1.6,
            joint_size=30.0,
        )
        if mesh_vertices is not None:
            points_for_limit.append(mesh_vertices[:, list(dims)])
        points_for_limit.append(pred_2d)
    if mode in {"gt", "overlay"} and _valid_j3(gt_kps_world):
        _plot_mesh_2d_dims(ax, gt_mesh_vertices, faces, dims=dims, color=gt_mesh_color, alpha=0.28)
        gt_2d = gt_kps_world[:, list(dims)]
        _draw_skeleton_2d(
            ax,
            gt_2d,
            bone_color=gt_bone_color,
            joint_color=gt_joint_color,
            linewidth=1.5,
            joint_size=28.0,
        )
        if gt_mesh_vertices is not None:
            points_for_limit.append(gt_mesh_vertices[:, list(dims)])
        points_for_limit.append(gt_2d)
    _set_2d_limits(ax, points_for_limit)
    if invert_y:
        ax.invert_yaxis()
    _clean_axis_2d(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def visualize_and_save_global(
    pred_file: str,
    data_root: str,
    save_dir: str,
    start: int,
    end: int,
    dataset: str,
    pose_encoding_type: str,
    pred_keypoints_key: str,
    gt_keypoints_key: str,
    smpl_params_key: str,
    smpl_model_path: str,
    lidar_frame_mode: str,
    sensor_modality: str,
    sensor_idx: int,
    max_pc_points: int,
    skl_format: str,
):
    os.makedirs(save_dir, exist_ok=True)
    data = load(pred_file)
    sample_ids = data.get("sample_ids", None)
    if sample_ids is None:
        raise KeyError("Prediction file must contain `sample_ids`.")
    total = len(sample_ids)
    if total == 0:
        raise ValueError("Empty `sample_ids` in prediction file.")
    if dataset != "humman_preproc":
        raise NotImplementedError(f"Unsupported dataset `{dataset}`. Currently only `humman_preproc` is supported.")

    pred_all = data.get(pred_keypoints_key, None)
    gt_all = data.get(gt_keypoints_key, None)
    smpl_params_all = data.get(smpl_params_key, None)
    gt_smpl_params_all = data.get("gt_smpl_params", None)
    pred_smpl_kps_all = data.get("pred_smpl_keypoints", None)
    if pred_all is None and pred_smpl_kps_all is not None:
        pred_all = pred_smpl_kps_all
    if pred_all is None:
        raise KeyError(
            f"Missing `{pred_keypoints_key}` and fallback `pred_smpl_keypoints` in prediction file."
        )

    start = int(start)
    end = int(end)
    if start < 0:
        raise ValueError(f"`--start` must be >= 0, got {start}.")
    if end == -1:
        end = total - 1
    elif end < -1:
        raise ValueError(f"`--end` must be >= 0 or -1 (use -1 for last sample), got {end}.")
    if start >= total:
        raise ValueError(f"`--start`={start} out of range for total samples={total}.")
    if end >= total:
        raise ValueError(f"`--end`={end} out of range for total samples={total}.")
    if start > end:
        raise ValueError(f"`--start` must be <= `--end`, got start={start}, end={end}.")

    chosen = list(range(start, end + 1))

    smpl_model = SMPL(model_path=smpl_model_path)
    faces = smpl_model.th_faces.detach().cpu().numpy()
    pred_bone_color = "#D55E00"
    pred_joint_color = "#A73E00"
    gt_bone_color = "#0072B2"
    gt_joint_color = "#004B78"
    pred_mesh_color = "#F4C095"
    gt_mesh_color = "#9CC4E4"
    camera_cache: Dict[str, Dict[str, dict]] = {}

    for idx in tqdm(chosen, desc="visualize", total=len(chosen)):
        sample_id = sample_ids[idx]
        try:
            rgb, pc_world, rgb_src_path = _load_humman_inputs(
                sample_id=sample_id,
                data_root=data_root,
                max_pc_points=max_pc_points,
            )
        except Exception as exc:
            print(f"[WARN] skip sample {sample_id}: failed to load input data ({exc})")
            continue

        try:
            gt_world_extr = _load_world_extrinsic_from_humman_camera(
                sample_id=sample_id,
                data_root=data_root,
                modality=sensor_modality,
                camera_cache=camera_cache,
            )
        except Exception as exc:
            print(
                f"[WARN] skip sample {sample_id}: failed to load raw {sensor_modality} camera "
                f"from `{data_root}/cameras` ({exc})"
            )
            continue

        pred_kps = _to_numpy_float(_get_sample_keypoints(pred_all, idx))
        if not _valid_j3(pred_kps):
            print(f"[WARN] skip sample {sample_id}: invalid predicted keypoints shape.")
            continue
        pred_kps_world = _transform_points(pred_kps, gt_world_extr)
        sample_skl_format = str(skl_format).lower()
        if sample_skl_format == "auto":
            sample_skl_format = _infer_skl_format(int(pred_kps_world.shape[0]))

        gt_kps = _to_numpy_float(_get_sample_keypoints(gt_all, idx))
        if _valid_j3(gt_kps):
            gt_kps_world = _transform_points(gt_kps, gt_world_extr)
        else:
            gt_kps_world = None

        pred_smpl_kps = _to_numpy_float(_get_sample_keypoints(pred_smpl_kps_all, idx))
        if _valid_j3(pred_smpl_kps):
            pred_smpl_kps_world = _transform_points(pred_smpl_kps, gt_world_extr)
        else:
            pred_smpl_kps_world = None

        smpl_params = _to_numpy_float(_extract_sample_any(smpl_params_all, idx))
        mesh_vertices, mesh_joints = _load_smpl_vertices(smpl_model, smpl_params)
        if mesh_vertices is not None:
            mesh_vertices = _transform_points(mesh_vertices, gt_world_extr)
        if mesh_joints is not None:
            mesh_joints = _transform_points(mesh_joints, gt_world_extr)
        mesh_anchor_kps = pred_smpl_kps_world if pred_smpl_kps_world is not None else pred_kps_world
        mesh_vertices, mesh_joints = _align_mesh_to_keypoints(
            mesh_vertices, mesh_joints, mesh_anchor_kps, sample_skl_format
        )

        gt_smpl_params = _to_numpy_float(_extract_sample_any(gt_smpl_params_all, idx))
        gt_mesh_vertices, gt_mesh_joints = _load_smpl_vertices(smpl_model, gt_smpl_params)
        if gt_mesh_vertices is not None:
            gt_mesh_vertices = _transform_points(gt_mesh_vertices, gt_world_extr)
        if gt_mesh_joints is not None:
            gt_mesh_joints = _transform_points(gt_mesh_joints, gt_world_extr)
        gt_mesh_vertices, gt_mesh_joints = _align_mesh_to_keypoints(
            gt_mesh_vertices, gt_mesh_joints, gt_kps_world, sample_skl_format
        )

        sample_dir = osp.join(save_dir, _to_save_name(sample_id))
        os.makedirs(sample_dir, exist_ok=True)

        try:
            shutil.copy2(rgb_src_path, osp.join(sample_dir, "rgb.jpg"))
        except Exception:
            cv2.imwrite(osp.join(sample_dir, "rgb.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        for mode_name in ("pred", "gt", "overlay"):
            _save_world_3d(
                out_path=osp.join(sample_dir, f"world_3d_{mode_name}.png"),
                mode=mode_name,
                pc_world=pc_world,
                pred_kps_world=pred_kps_world,
                gt_kps_world=gt_kps_world,
                mesh_vertices=mesh_vertices,
                gt_mesh_vertices=gt_mesh_vertices,
                faces=faces,
                pred_bone_color=pred_bone_color,
                pred_joint_color=pred_joint_color,
                gt_bone_color=gt_bone_color,
                gt_joint_color=gt_joint_color,
                pred_mesh_color=pred_mesh_color,
                gt_mesh_color=gt_mesh_color,
            )
            _save_world_2d_view(
                out_path=osp.join(sample_dir, f"world_xy_{mode_name}.png"),
                mode=mode_name,
                dims=(0, 1),
                invert_y=True,
                pred_kps_world=pred_kps_world,
                gt_kps_world=gt_kps_world,
                mesh_vertices=mesh_vertices,
                gt_mesh_vertices=gt_mesh_vertices,
                faces=faces,
                pred_bone_color=pred_bone_color,
                pred_joint_color=pred_joint_color,
                gt_bone_color=gt_bone_color,
                gt_joint_color=gt_joint_color,
                pred_mesh_color=pred_mesh_color,
                gt_mesh_color=gt_mesh_color,
            )
            _save_world_2d_view(
                out_path=osp.join(sample_dir, f"world_xz_{mode_name}.png"),
                mode=mode_name,
                dims=(0, 2),
                invert_y=False,
                pred_kps_world=pred_kps_world,
                gt_kps_world=gt_kps_world,
                mesh_vertices=mesh_vertices,
                gt_mesh_vertices=gt_mesh_vertices,
                faces=faces,
                pred_bone_color=pred_bone_color,
                pred_joint_color=pred_joint_color,
                gt_bone_color=gt_bone_color,
                gt_joint_color=gt_joint_color,
                pred_mesh_color=pred_mesh_color,
                gt_mesh_color=gt_mesh_color,
            )
            _save_world_2d_view(
                out_path=osp.join(sample_dir, f"world_yz_{mode_name}.png"),
                mode=mode_name,
                dims=(1, 2),
                invert_y=False,
                pred_kps_world=pred_kps_world,
                gt_kps_world=gt_kps_world,
                mesh_vertices=mesh_vertices,
                gt_mesh_vertices=gt_mesh_vertices,
                faces=faces,
                pred_bone_color=pred_bone_color,
                pred_joint_color=pred_joint_color,
                gt_bone_color=gt_bone_color,
                gt_joint_color=gt_joint_color,
                pred_mesh_color=pred_mesh_color,
                gt_mesh_color=gt_mesh_color,
            )

    print(f"Saved {len(chosen)} visualization images to: {save_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize predictions by mapping keypoints/SMPL from sensor to global space."
    )
    parser.add_argument("--pred-file", type=str, required=True, help="Path to prediction pkl file.")
    parser.add_argument("--data-root", type=str, default="/root/autodl-tmp/humman_cropped")
    parser.add_argument("--save-dir", type=str, default="visualization_results_global")
    parser.add_argument("--dataset", type=str, default="humman_preproc", choices=["humman_preproc"])
    parser.add_argument("--start", type=int, default=0, help="Start sample index (inclusive).")
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="End sample index (inclusive). Use -1 for the last sample.",
    )
    parser.add_argument("--pred-keypoints-key", type=str, default="pred_keypoints")
    parser.add_argument("--gt-keypoints-key", type=str, default="gt_keypoints")
    parser.add_argument("--smpl-params-key", type=str, default="pred_smpl_params")
    parser.add_argument(
        "--pose-encoding-type",
        type=str,
        default="absT_quaR_FoV",
        help="Unused in raw-camera mode; kept for CLI compatibility.",
    )
    parser.add_argument(
        "--lidar-frame-mode",
        type=str,
        default="seq_fixed",
        choices=["seq_fixed", "per_sample"],
        help="Unused in raw-camera mode; kept for CLI compatibility.",
    )
    parser.add_argument("--sensor-modality", type=str, default="lidar", choices=["lidar", "depth", "rgb"])
    parser.add_argument("--sensor-idx", type=int, default=0)
    parser.add_argument("--smpl-model-path", type=str, default="weights/smpl/SMPL_NEUTRAL.pkl")
    parser.add_argument(
        "--skl-format",
        type=str,
        default="auto",
        choices=["auto", "smpl", "coco", "panoptic_coco19", "simple_coco"],
        help="Skeleton format for mesh-to-keypoint alignment. `auto` infers from joint count.",
    )
    parser.add_argument("--max-pc-points", type=int, default=60000)
    return parser


def main():
    args = build_parser().parse_args()
    visualize_and_save_global(
        pred_file=args.pred_file,
        data_root=args.data_root,
        save_dir=args.save_dir,
        start=args.start,
        end=args.end,
        dataset=args.dataset,
        pose_encoding_type=args.pose_encoding_type,
        pred_keypoints_key=args.pred_keypoints_key,
        gt_keypoints_key=args.gt_keypoints_key,
        smpl_params_key=args.smpl_params_key,
        smpl_model_path=args.smpl_model_path,
        lidar_frame_mode=args.lidar_frame_mode,
        sensor_modality=args.sensor_modality,
        sensor_idx=args.sensor_idx,
        max_pc_points=args.max_pc_points,
        skl_format=args.skl_format,
    )


if __name__ == "__main__":
    main()
