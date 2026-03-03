#!/usr/bin/env python3
"""Run SAM-3D-Body inference on a config-selected sample and log to rerun."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from misc.utils import load_cfg, merge_args_cfg
from misc.registry import create_dataset
from misc.skeleton import SMPLSkeleton
from rerun_utils.camera import resolve_view_extrinsics, transform_points_to_camera
from rerun_utils.geometry import align_keypoints_to_joints_scale, rotate_points_180_y
from rerun_utils.image import process_image_for_display
from rerun_utils.layout import (
    build_input_layout_from_config,
    create_rerun_blueprint,
    split_sample_views,
)
from rerun_utils.logging3d import log_mesh_3d, log_skeleton_views
from rerun_utils.session import init_rerun_session, init_world_axes, set_frame_timeline
from rerun_utils.smpl import load_smpl_model, smpl_params_to_mesh, split_smpl_params
from rerun_utils.temporal import select_sample_frame_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM-3D-Body rerun visualization")
    parser.add_argument("-c", "--cfg", required=True, help="Config path")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index in selected split")
    parser.add_argument("--frame-index", type=int, default=-1, help="-1 means center frame")
    parser.add_argument("--num-frames", type=int, default=1, help="Number of frames to visualize from sample window")
    parser.add_argument("--device", default="cuda", help="Inference device")
    parser.add_argument(
        "--smpl-model",
        default=None,
        help="Optional SMPL model path for GT mesh logging. Defaults to config `smpl_model_path` "
        "or `weights/smpl/SMPL_NEUTRAL.pkl` when available.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="/opt/data/SAM_3dbody_checkpoints",
        help="SAM-3D-Body checkpoint root",
    )
    parser.add_argument(
        "--render-mode",
        default="auto",
        choices=["mesh", "overlay", "auto"],
        help="Force mesh, force overlay, or auto-fallback",
    )
    parser.add_argument("--recording_name", default="sam3d_body_inference")
    parser.add_argument("--save_rrd", default=None, help="Save recording path")
    parser.add_argument("--no_serve", action="store_true", help="Disable live server")
    parser.add_argument("--web_port", type=int, default=9090)
    parser.add_argument("--grpc_port", type=int, default=9091)
    parser.add_argument(
        "--debug-image-stats",
        action="store_true",
        help="Print RGB image stats (shape/dtype/min/max) before rerun logging.",
    )
    parser.add_argument(
        "--gt-coordinate-space",
        default=None,
        help="Ground-truth 3D coordinate space: `canonical` or `camera`. "
        "When omitted, uses `gt_coordinate_space` from config or `canonical`.",
    )
    return parser.parse_args()


def _resolve_dataset_cfg(hparams, split: str):
    if split == "train":
        return hparams.train_dataset, hparams.train_pipeline
    if split == "val":
        return hparams.val_dataset, hparams.val_pipeline
    return hparams.test_dataset, hparams.test_pipeline


def _check_checkpoint_paths(checkpoint_root: Path) -> tuple[Path, Path, Path]:
    cfg_path = checkpoint_root / "model_config.yaml"
    ckpt_path = checkpoint_root / "model.ckpt"
    # Prefer assets layout for mhr_model.pt, keep root-level fallback for compatibility.
    mhr_path = checkpoint_root / "assets" / "mhr_model.pt"
    if not mhr_path.exists():
        mhr_path = checkpoint_root / "mhr_model.pt"
    required = [cfg_path, ckpt_path, mhr_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required SAM-3D-Body file(s): "
            + ", ".join(missing)
            + ". Expected files under /opt/data/SAM_3dbody_checkpoints/ (mhr_model.pt in assets/ by default)."
        )
    return cfg_path, ckpt_path, mhr_path


def _coerce_rgb_for_rerun(image: np.ndarray) -> np.ndarray:
    """Normalize image to uint8 RGB for rerun logging."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (H,W,3), got {arr.shape}")
    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32)
    finite_mask = np.isfinite(arr)
    if not finite_mask.all():
        arr = np.where(finite_mask, arr, 0.0)

    min_val = float(arr.min())
    max_val = float(arr.max())
    if min_val >= 0.0 and max_val <= 1.0 + 1e-6:
        arr = arr * 255.0
    elif min_val < 0.0:
        # Assume normalized image and map to display range.
        arr = (arr - min_val) / (max_val - min_val + 1e-6) * 255.0

    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def _extract_rgb_sequence(image_data, denorm_params: dict | None) -> np.ndarray:
    frame = process_image_for_display(
        image_data,
        denorm_params=denorm_params,
        key="rgb",
        keep_temporal=True,
    )
    arr = np.asarray(frame)
    if arr.ndim == 3:
        return _coerce_rgb_for_rerun(arr)[None, ...]
    if arr.ndim == 4:
        return np.stack([_coerce_rgb_for_rerun(arr[i]) for i in range(arr.shape[0])], axis=0)
    raise ValueError(f"Expected RGB sequence with 3/4 dims, got {arr.shape}")


def _log_image_stats(tag: str, image: np.ndarray) -> None:
    print(
        f"[image-debug] {tag}: shape={tuple(image.shape)} dtype={image.dtype} "
        f"min={float(image.min()):.3f} max={float(image.max()):.3f}"
    )


def _side_view_transform(points: np.ndarray) -> np.ndarray:
    """Apply the same side transform used by mesh and keypoints."""
    side_points = points.copy()
    side_points[..., [0, 2]] = side_points[..., [2, 0]]
    side_points[..., 0] *= -1.0
    return side_points


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _sam3d_skeleton() -> tuple[list[tuple[int, int]], list[list[int]]]:
    """Load SAM-3D-Body mhr70 skeleton edges and colors."""
    from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info

    keypoint_info = mhr70_pose_info["keypoint_info"]
    skeleton_info = mhr70_pose_info["skeleton_info"]
    name_to_id = {item["name"]: int(kpt_id) for kpt_id, item in keypoint_info.items()}
    edges: list[tuple[int, int]] = []
    colors: list[list[int]] = []
    for _, sk in skeleton_info.items():
        src_name, dst_name = sk["link"]
        if src_name in name_to_id and dst_name in name_to_id:
            edges.append((name_to_id[src_name], name_to_id[dst_name]))
            colors.append(sk.get("color", [96, 96, 255]))
    return edges, colors


def _draw_overlay(
    image_rgb: np.ndarray,
    outputs: list[dict],
    skeleton_edges: list[tuple[int, int]],
    line_color: tuple[int, int, int] = (255, 128, 0),
) -> np.ndarray:
    canvas = image_rgb.copy()
    for person in outputs:
        bbox = np.asarray(person.get("bbox", []))
        if bbox.shape == (4,):
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

        keypoints_2d = person.get("pred_keypoints_2d")
        if keypoints_2d is not None:
            kpts = np.asarray(keypoints_2d, dtype=np.float32)
            for i, j in skeleton_edges:
                if i < len(kpts) and j < len(kpts):
                    x1, y1 = int(kpts[i, 0]), int(kpts[i, 1])
                    x2, y2 = int(kpts[j, 0]), int(kpts[j, 1])
                    cv2.line(canvas, (x1, y1), (x2, y2), line_color, 1, cv2.LINE_AA)

            for x, y, *_ in kpts:
                cv2.circle(canvas, (int(x), int(y)), 2, (255, 0, 0), -1)
    return canvas


def _load_sam3d_estimator(checkpoint_root: Path, device: str):
    _, ckpt_path, mhr_path = _check_checkpoint_paths(checkpoint_root)

    sam3d_root = REPO_ROOT / "third_party" / "sam-3d-body"
    sys.path.insert(0, str(sam3d_root))
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    model, model_cfg = load_sam_3d_body(
        checkpoint_path=str(ckpt_path),
        device=device,
        mhr_path=str(mhr_path),
    )
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )
    return estimator


def _select_temporal_frame(arr: np.ndarray, frame_idx: int) -> np.ndarray:
    if arr.ndim == 0:
        return arr
    if arr.ndim >= 2 and arr.shape[0] > frame_idx:
        # Temporal tensors are expected to have T in dim 0 for this script.
        return arr[frame_idx]
    return arr


def _log_gt_outputs(
    sample: dict,
    smpl_model,
    faces: np.ndarray | None,
    source_frame_idx: int,
    gt_coordinate_space: str,
    view_extrinsic: np.ndarray | None,
) -> tuple[bool, bool]:
    gt_kpts_ok = False
    gt_mesh_ok = False
    gt_joints_front = None

    gt_keypoints_data = None
    if "gt_keypoints_seq" in sample:
        gt_keypoints_data = _to_numpy(sample["gt_keypoints_seq"]).astype(np.float32)
    elif "gt_keypoints" in sample:
        gt_keypoints_data = _to_numpy(sample["gt_keypoints"]).astype(np.float32)

    has_smpl_model = smpl_model is not None

    gt_smpl_params = None
    if "gt_smpl_params_seq" in sample or "gt_smpl_params" in sample:
        gt_params = _to_numpy(sample.get("gt_smpl_params_seq", sample.get("gt_smpl_params"))).astype(np.float32)
        if gt_params.ndim >= 2:
            gt_params = _select_temporal_frame(gt_params, source_frame_idx)
        gt_smpl_params = split_smpl_params(gt_params)
    elif all(k in sample for k in ("gt_global_orient", "gt_body_pose", "gt_betas", "gt_transl")):
        gt_global_orient = _to_numpy(sample["gt_global_orient"]).astype(np.float32)
        gt_body_pose = _to_numpy(sample["gt_body_pose"]).astype(np.float32)
        gt_betas = _to_numpy(sample["gt_betas"]).astype(np.float32)
        gt_transl = _to_numpy(sample["gt_transl"]).astype(np.float32)
        if gt_global_orient.ndim >= 2:
            gt_global_orient = _select_temporal_frame(gt_global_orient, source_frame_idx)
        if gt_body_pose.ndim >= 2:
            gt_body_pose = _select_temporal_frame(gt_body_pose, source_frame_idx)
        if gt_betas.ndim >= 2:
            gt_betas = _select_temporal_frame(gt_betas, source_frame_idx)
        if gt_transl.ndim >= 2:
            gt_transl = _select_temporal_frame(gt_transl, source_frame_idx)
        gt_smpl_params = {
            "global_orient": gt_global_orient,
            "body_pose": gt_body_pose,
            "betas": gt_betas,
            "transl": gt_transl,
        }

    if has_smpl_model and gt_smpl_params is not None:
        try:
            gt_vertices, gt_joints, gt_faces = smpl_params_to_mesh(smpl_model, gt_smpl_params, device="cpu")
            gt_vertices = gt_vertices[0]
            gt_joints = gt_joints[0]
            if gt_coordinate_space == "camera":
                if view_extrinsic is None:
                    raise ValueError(
                        "GT camera-coordinate mode requested but no per-view extrinsic is available for GT mesh."
                    )
                gt_vertices = transform_points_to_camera(gt_vertices, view_extrinsic)
                gt_joints = transform_points_to_camera(gt_joints, view_extrinsic)
            gt_vertices = rotate_points_180_y(gt_vertices)
            gt_joints_front = rotate_points_180_y(gt_joints)
            log_mesh_3d("world/front/ground_truth/mesh", gt_vertices, gt_faces, color=(100, 255, 150), alpha=0.55)
            log_mesh_3d(
                "world/side/ground_truth/mesh",
                _side_view_transform(gt_vertices),
                gt_faces,
                color=(100, 255, 150),
                alpha=0.55,
            )
            gt_mesh_ok = True
        except Exception as exc:  # noqa: BLE001
            rr.log("world/info/gt_mesh_error", rr.TextLog(str(exc)))

    if gt_keypoints_data is not None:
        gt_keypoints = gt_keypoints_data
        if gt_keypoints.ndim == 3:
            gt_keypoints = _select_temporal_frame(gt_keypoints, source_frame_idx)
        if gt_coordinate_space == "camera":
            if view_extrinsic is None:
                raise ValueError(
                    "GT camera-coordinate mode requested but no per-view extrinsic is available."
                )
            gt_keypoints = transform_points_to_camera(gt_keypoints, view_extrinsic)
        gt_keypoints = rotate_points_180_y(gt_keypoints)
        if gt_joints_front is not None and gt_keypoints.shape[0] == gt_joints_front.shape[0]:
            gt_keypoints = align_keypoints_to_joints_scale(gt_keypoints, gt_joints_front)
        log_skeleton_views(
            "ground_truth",
            gt_keypoints,
            SMPLSkeleton,
            color=(0, 255, 100),
            radius=0.02,
            side_transform=_side_view_transform,
        )
        gt_kpts_ok = True

    return gt_kpts_ok, gt_mesh_ok


def _log_sam3d_outputs(
    outputs: list[dict],
    estimator,
    render_mode: str,
    skeleton_edges: list[tuple[int, int]],
    skeleton_edge_colors: list[list[int]],
) -> str:
    if not outputs:
        rr.log("world/info/num_people", rr.TextLog("0"))
        return "none"

    rr.log("world/info/num_people", rr.TextLog(str(len(outputs))))
    person = outputs[0]
    can_render_mesh = "pred_vertices" in person and "pred_cam_t" in person and hasattr(estimator, "faces")

    if render_mode == "overlay":
        return "overlay"
    if render_mode == "mesh" and not can_render_mesh:
        raise RuntimeError("render-mode=mesh requested but SAM-3D-Body output has no mesh fields.")
    if render_mode == "auto" and not can_render_mesh:
        return "overlay"

    pred_cam_t = np.asarray(person["pred_cam_t"], dtype=np.float32).reshape(1, 3)
    vertices_cam = np.asarray(person["pred_vertices"], dtype=np.float32) + pred_cam_t
    vertices_front = rotate_points_180_y(vertices_cam)
    faces = np.asarray(estimator.faces, dtype=np.int32)
    log_mesh_3d("world/front/prediction/mesh", vertices_front, faces, color=(100, 180, 255), alpha=0.65)
    side_vertices = _side_view_transform(vertices_front)
    log_mesh_3d("world/side/prediction/mesh", side_vertices, faces, color=(100, 180, 255), alpha=0.65)

    if "pred_keypoints_3d" in person:
        keypoints_cam = np.asarray(person["pred_keypoints_3d"], dtype=np.float32) + pred_cam_t
        keypoints_front = rotate_points_180_y(keypoints_cam)
        log_skeleton_views(
            "prediction",
            keypoints_front,
            SMPLSkeleton,
            color=(0, 200, 255),
            radius=0.02,
            bones=skeleton_edges,
            bone_colors=skeleton_edge_colors,
            side_transform=_side_view_transform,
        )

    return "mesh"


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    if args.num_frames < 1:
        raise ValueError(f"--num-frames must be >= 1, got {args.num_frames}")

    cfg = load_cfg(args.cfg)
    class MockArgs:
        pass

    mock_args = MockArgs()
    mock_args.checkpoint_path = ""
    mock_args.gpus = 1
    mock_args.num_workers = 0
    mock_args.batch_size = 1
    mock_args.batch_size_eva = 1
    mock_args.pin_memory = False
    mock_args.prefetch_factor = 2
    mock_args.use_wandb = False
    mock_args.save_test_preds = False
    hparams = merge_args_cfg(mock_args, cfg)
    denorm_params = getattr(hparams, "vis_denorm_params", None)
    gt_coordinate_space_cfg = str(getattr(hparams, "gt_coordinate_space", "canonical"))
    gt_coordinate_space = (
        str(args.gt_coordinate_space) if args.gt_coordinate_space is not None else gt_coordinate_space_cfg
    )
    gt_coordinate_space = gt_coordinate_space.lower().strip()
    if gt_coordinate_space not in {"canonical", "camera"}:
        raise ValueError(
            f"Invalid GT coordinate space `{gt_coordinate_space}`. "
            "Use `canonical` or `camera` via `--gt-coordinate-space` "
            "or config key `gt_coordinate_space`."
        )
    # Single mode per run for clarity and deterministic comparisons.
    assert isinstance(gt_coordinate_space, str)

    dataset_cfg, pipeline_cfg = _resolve_dataset_cfg(hparams, args.split)
    dataset, _ = create_dataset(dataset_cfg["name"], dataset_cfg["params"], pipeline_cfg)
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset split `{args.split}` is empty.")
    sample_idx = max(0, min(args.sample_idx, len(dataset) - 1))
    anchor_sample = dataset[sample_idx]

    input_layout = build_input_layout_from_config(dataset_cfg["params"])
    blueprint = create_rerun_blueprint(input_layout)
    expected_views = {item["modality"]: item["num_views"] for item in input_layout}

    estimator = _load_sam3d_estimator(Path(args.checkpoint_root), args.device)
    sam_edges, sam_edge_colors = _sam3d_skeleton()

    smpl_model = None
    smpl_model_path = args.smpl_model or getattr(hparams, "smpl_model_path", None)
    if smpl_model_path is None:
        default_smpl = REPO_ROOT / "weights" / "smpl" / "SMPL_NEUTRAL.pkl"
        if default_smpl.exists():
            smpl_model_path = str(default_smpl)
    if smpl_model_path:
        try:
            smpl_model = load_smpl_model(smpl_model_path, device="cpu")
        except Exception as exc:  # noqa: BLE001
            print(f"[SAM3D-RERUN] WARN: failed to load SMPL model for GT mesh logging ({smpl_model_path}): {exc}")
    else:
        print("[SAM3D-RERUN] WARN: no SMPL model path found; GT mesh logging is disabled.")

    init_rerun_session(
        recording_name=args.recording_name,
        save_rrd=args.save_rrd,
        no_serve=args.no_serve,
        web_port=args.web_port,
        grpc_port=args.grpc_port,
    )
    rr.send_blueprint(blueprint)
    # SAM-3D-Body outputs are interpreted in camera-like coordinates (x right, y down, z forward).
    # Use Y-down view coordinates in rerun to avoid vertical inversion in 3D views.
    init_world_axes(rr.ViewCoordinates.RIGHT_HAND_Y_DOWN)

    if "input_rgb" not in anchor_sample:
        raise ValueError("Selected sample does not contain `input_rgb`; SAM-3D-Body requires RGB input.")

    anchor_rgb_views = split_sample_views(
        anchor_sample["input_rgb"],
        expected_views=expected_views.get("rgb"),
        modality="rgb",
    )
    anchor_rgb_sequences = [_extract_rgb_sequence(view, denorm_params=denorm_params) for view in anchor_rgb_views]
    center_view_idx = len(anchor_rgb_views) // 2
    seq_lengths = [seq.shape[0] for seq in anchor_rgb_sequences]
    if len(set(seq_lengths)) != 1:
        raise ValueError(f"Inconsistent RGB temporal lengths across views: {seq_lengths}")
    anchor_temporal_len = seq_lengths[0]
    steps = select_sample_frame_steps(
        dataset_size=len(dataset),
        sample_idx=sample_idx,
        temporal_len=anchor_temporal_len,
        frame_index=args.frame_index,
        num_frames=args.num_frames,
    )
    if len(steps) == 0:
        raise RuntimeError("No visualization steps were generated from the provided frame arguments.")
    rr.log("world/info/num_visualized_frames", rr.TextLog(str(len(steps))))
    rr.log("world/info/gt_coordinate_space", rr.TextLog(gt_coordinate_space))

    cached_step_idx = None
    cached_sample = None
    cached_rgb_sequences = None
    cached_view_extrinsics = None

    for local_idx, (step_sample_idx, source_frame_idx) in enumerate(steps):
        if cached_step_idx != step_sample_idx:
            cached_sample = dataset[step_sample_idx]
            if "input_rgb" not in cached_sample:
                raise ValueError(
                    f"Sample idx {step_sample_idx} does not contain `input_rgb`; SAM-3D-Body requires RGB input."
                )
            rgb_views = split_sample_views(
                cached_sample["input_rgb"],
                expected_views=expected_views.get("rgb"),
                modality="rgb",
            )
            cached_rgb_sequences = [_extract_rgb_sequence(view, denorm_params=denorm_params) for view in rgb_views]
            if gt_coordinate_space == "camera":
                cached_view_extrinsics = resolve_view_extrinsics(
                    cached_sample.get("rgb_camera"),
                    num_rgb_views=len(rgb_views),
                )
            else:
                cached_view_extrinsics = None
            cached_step_idx = step_sample_idx

        sample_id = str(cached_sample.get("sample_id", f"idx_{step_sample_idx}"))
        set_frame_timeline(local_idx, sample_id=sample_id)
        rr.log("world/info/source_frame_index", rr.TextLog(str(source_frame_idx)))
        rr.log("world/info/source_sample_index", rr.TextLog(str(step_sample_idx)))

        for view_idx, rgb_seq in enumerate(cached_rgb_sequences):
            safe_idx = max(0, min(source_frame_idx, rgb_seq.shape[0] - 1))
            rgb_image = rgb_seq[safe_idx]
            if args.debug_image_stats:
                _log_image_stats(f"sample_{step_sample_idx}/frame_{safe_idx}/rgb/view_{view_idx}", rgb_image)
            rr.log(f"world/inputs/rgb/view_{view_idx}/image", rr.Image(rgb_image))

            if view_idx == center_view_idx:
                outputs = estimator.process_one_image(rgb_image)
                overlay = _draw_overlay(rgb_image, outputs, skeleton_edges=sam_edges)
                if args.debug_image_stats:
                    _log_image_stats(f"sample_{step_sample_idx}/frame_{safe_idx}/rgb/view_{view_idx}/overlay", overlay)
                rr.log(f"world/inputs/rgb/view_{view_idx}/overlay", rr.Image(overlay))
                actual_mode = _log_sam3d_outputs(
                    outputs,
                    estimator,
                    args.render_mode,
                    skeleton_edges=sam_edges,
                    skeleton_edge_colors=sam_edge_colors,
                )
                rr.log("world/info/render_mode", rr.TextLog(actual_mode))
                gt_keypoints_ok, gt_mesh_ok = _log_gt_outputs(
                    cached_sample,
                    smpl_model,
                    np.asarray(estimator.faces, dtype=np.int32),
                    source_frame_idx=safe_idx,
                    gt_coordinate_space=gt_coordinate_space,
                    view_extrinsic=None if cached_view_extrinsics is None else cached_view_extrinsics[view_idx],
                )
                rr.log("world/info/gt_keypoints_available", rr.TextLog(str(gt_keypoints_ok)))
                rr.log("world/info/gt_mesh_available", rr.TextLog(str(gt_mesh_ok)))

    print(f"[SAM3D-RERUN] split={args.split} sample_idx={sample_idx} steps={len(steps)}")
    if args.save_rrd:
        print(f"[SAM3D-RERUN] saved: {args.save_rrd}")
    if not args.no_serve:
        print(f"[SAM3D-RERUN] web viewer: http://localhost:{args.web_port}")


if __name__ == "__main__":
    main()
