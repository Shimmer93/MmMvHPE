#!/usr/bin/env python3
"""
Inference Visualization with Rerun

This script runs inference with a trained model on multimodal input data
and visualizes the results (keypoints, SMPL mesh) using Rerun.

Compatible with any config file (humman_ours.yml, h36m_*, etc.)

Usage:
    python scripts/visualize_inference_rerun.py \
        --cfg configs/dev/humman_ours.yml \
        --checkpoint logs/dev/ours20251220_224245/last.ckpt \
        --num_samples 50 \
        --device cuda
"""
import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rerun as rr

from misc.utils import load_cfg, merge_args_cfg, torch2numpy
from misc.pose_enc import pose_encoding_to_extri_intri
from misc.registry import create_dataset
from models.model_api import LitModel
from rerun_utils.camera import resolve_reference_extrinsic, transform_points_to_camera
from rerun_utils.geometry import (
    align_keypoints_to_joints_scale,
    rotate_points_180_y,
    rotate_points_x,
)
from rerun_utils.image import process_image_for_display
from rerun_utils.layout import (
    build_input_layout_from_config,
    create_rerun_blueprint,
    split_sample_views,
)
from rerun_utils.logging3d import (
    log_point_cloud_3d,
    log_skeleton_views,
    log_smpl_views,
)
from rerun_utils.session import init_rerun_session, init_world_axes, set_frame_timeline
from rerun_utils.smpl import (
    canonicalize_smpl_params,
    get_skeleton_class,
    load_smpl_model,
    smpl_params_to_mesh,
    split_smpl_params,
)
from rerun_utils.temporal import select_frame_indices, select_sample_frame_steps




def run_inference(model, batch, device):
    """Run model inference on a batch.
    
    Args:
        model: LitModel instance
        batch: Data batch dict
        device: Device to run on
        
    Returns:
        pred_dict: Dictionary with predictions
    """
    # Move batch to device
    batch_device = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_device[k] = v.to(device)
        else:
            batch_device[k] = v
    
    model.eval()
    with torch.no_grad():
        # Extract and aggregate features
        feats = model.extract_features(batch_device)
        feats_agg = model.aggregate_features(feats, batch_device)
        
        # Get predictions from heads
        pred_dict = {}
        if model.with_keypoint_head:
            pred_dict['pred_keypoints'] = model.keypoint_head.predict(feats_agg)
        if model.with_smpl_head:
            pred_dict['pred_smpl'] = model.smpl_head.predict(feats_agg)
        if model.with_camera_head:
            pred_dict['pred_cameras'] = model.camera_head.predict(
                feats_agg,
                data_batch=batch_device,
                pred_dict=pred_dict,
            )
    
    return pred_dict


def _resolve_predicted_extrinsic(pred_dict, sample, sensor_label: str, image_hw: tuple[int, int]):
    pred_cameras = pred_dict.get("pred_cameras", None)
    if pred_cameras is None:
        return None
    if not isinstance(pred_cameras, torch.Tensor):
        return None
    if pred_cameras.dim() != 3 or pred_cameras.shape[0] < 1:
        return None

    modalities = sample.get("modalities", [])
    if isinstance(modalities, (list, tuple)) and modalities and isinstance(modalities[0], (list, tuple)):
        modalities = modalities[0]
    modalities = [str(m).lower() for m in modalities]

    label = str(sensor_label).lower()
    if label == "lidar":
        candidate_modalities = ["lidar", "depth"]
    else:
        candidate_modalities = [label]

    m_idx = None
    for name in candidate_modalities:
        if name in modalities:
            m_idx = modalities.index(name)
            break
    if m_idx is None or m_idx >= pred_cameras.shape[1]:
        return None

    enc = pred_cameras[:, m_idx : m_idx + 1, :]
    if not torch.isfinite(enc).all():
        return None

    extri, _ = pose_encoding_to_extri_intri(
        enc,
        image_size_hw=image_hw,
        pose_encoding_type="absT_quaR_FoV",
        build_intrinsics=False,
    )
    return extri[0, 0].detach().cpu().numpy().astype(np.float32)


def _expected_views_for_sample(sample: dict, modality: str, configured_views: int | None):
    selected = sample.get("selected_cameras", None)
    if isinstance(selected, dict):
        cams = selected.get(modality, None)
        if isinstance(cams, (list, tuple)) and len(cams) > 0:
            return len(cams)
    return configured_views


def _resolve_dataset_cfg(hparams, split: str):
    if split == "train":
        return hparams.train_dataset, hparams.train_pipeline
    if split == "val":
        return hparams.val_dataset, hparams.val_pipeline
    return hparams.test_dataset, hparams.test_pipeline


def _infer_temporal_length(sample: dict) -> int:
    for key in ("input_rgb", "input_depth", "input_lidar", "input_mmwave"):
        if key not in sample:
            continue
        value = sample[key]
        arr = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        if arr.ndim == 5:  # V,T,C,H,W or V,T,H,W,C
            return int(arr.shape[1])
        if arr.ndim == 4:  # T,C,H,W or T,H,W,C
            if arr.shape[1] in (1, 3, 4) or arr.shape[-1] in (1, 3, 4):
                return int(arr.shape[0])
            return int(arr.shape[1])
        if arr.ndim == 3:  # T,N,C or H,W,C
            if arr.shape[-1] in (1, 2, 3, 4) and arr.shape[0] > 8 and arr.shape[1] > 8:
                return 1
            return int(arr.shape[0])
    return 1


def _infer_image_hw(sample: dict) -> tuple[int, int]:
    if "input_rgb" in sample:
        rgb = sample["input_rgb"]
        arr = rgb.detach().cpu().numpy() if isinstance(rgb, torch.Tensor) else np.asarray(rgb)
        if arr.ndim == 5:  # V,T,C,H,W or V,T,H,W,C
            if arr.shape[-1] in (1, 3, 4):
                return int(arr.shape[-3]), int(arr.shape[-2])
            return int(arr.shape[-2]), int(arr.shape[-1])
        if arr.ndim == 4:  # T,C,H,W or T,H,W,C
            if arr.shape[-1] in (1, 3, 4):
                return int(arr.shape[-3]), int(arr.shape[-2])
            return int(arr.shape[-2]), int(arr.shape[-1])
        if arr.ndim == 3:  # C,H,W or H,W,C
            if arr.shape[-1] in (1, 3, 4):
                return int(arr.shape[0]), int(arr.shape[1])
            return int(arr.shape[1]), int(arr.shape[2])
    return (224, 224)


def _take_temporal(x, frame_idx: int):
    arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    if arr.ndim >= 3:
        tdim = 1 if (arr.ndim >= 4 and arr.shape[0] <= 4) else 0
        tlen = arr.shape[tdim]
        f = max(0, min(frame_idx, tlen - 1))
        return arr[:, f] if tdim == 1 else arr[f]
    return arr


def main():
    parser = argparse.ArgumentParser(
        description='Inference visualization with Rerun',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-c', '--cfg', type=str, required=True,
                        help='Path to config file (e.g., configs/dev/humman_ours.yml)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to use')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples to visualize')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting sample index')
    parser.add_argument('--sample-idx', type=int, default=None,
                        help='If set, visualize only this dataset sample index (overrides --start_idx/--num_samples).')
    parser.add_argument('--frame-index', type=int, default=-1,
                        help='Temporal frame index inside a sample window; -1 means center frame.')
    parser.add_argument('--num-frames', type=int, default=1,
                        help='Number of temporal frames to visualize per sample window.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on')
    parser.add_argument('--smpl_model', type=str,
                        default=None,
                        help='Path to SMPL model weights')
    parser.add_argument('--skeleton_format', type=str, default=None,
                        help='Skeleton format (smpl, h36m, coco). Auto-detected from config if not specified.')
    parser.add_argument('--recording_name', type=str, default='inference_vis',
                        help='Rerun recording name')
    parser.add_argument('--save_rrd', type=str, default=None,
                        help='Save Rerun recording to .rrd file')
    parser.add_argument('--web_port', type=int, default=9090,
                        help='Port for Rerun web viewer')
    parser.add_argument('--grpc_port', type=int, default=9091,
                        help='Port for Rerun gRPC server')
    parser.add_argument('--no_serve', action='store_true',
                        help='Disable live serving (only save to file). Faster when you only want .rrd output.')
    parser.add_argument('--show_gt', action='store_true', default=True,
                        help='Show ground truth alongside predictions')
    parser.add_argument('--no_mesh', action='store_true',
                        help='Disable mesh visualization (faster)')
    parser.add_argument('--canonical', action='store_true',
                        help='Visualize SMPL/keypoints in canonical space (zero global orient/transl).')
    parser.add_argument('--canonical_upright', action='store_true', default=True,
                        help='Rotate canonical visualization 180deg around X to avoid upside-down view.')
    parser.add_argument(
        '--coord-space',
        type=str,
        default='world',
        choices=['world', 'sensor'],
        help='3D visualization coordinate space.',
    )
    parser.add_argument(
        '--reference-sensor',
        type=str,
        default=None,
        choices=['rgb', 'depth', 'lidar'],
        help='Reference sensor label for sensor-space visualization.',
    )
    parser.add_argument(
        '--reference-view',
        type=int,
        default=None,
        help='Reference view index for sensor-space visualization.',
    )
    
    args = parser.parse_args()
    sensor_mode = args.coord_space == 'sensor'
    if sensor_mode:
        if args.reference_sensor is None or args.reference_view is None:
            raise ValueError(
                "Sensor mode requires both `--reference-sensor` and `--reference-view`."
            )
        if args.reference_view < 0:
            raise ValueError(f"`--reference-view` must be >= 0, got {args.reference_view}.")
        if args.canonical:
            raise ValueError("`--canonical` is not supported in sensor mode.")
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Load config
    print(f"Loading config from: {args.cfg}")
    cfg = load_cfg(args.cfg)
    
    # Create mock args for merging
    class MockArgs:
        pass
    mock_args = MockArgs()
    mock_args.checkpoint_path = args.checkpoint
    mock_args.gpus = 1
    mock_args.num_workers = 0
    mock_args.batch_size = 1
    mock_args.batch_size_eva = 1
    mock_args.pin_memory = False
    mock_args.prefetch_factor = 2
    mock_args.use_wandb = False
    mock_args.save_test_preds = False
    
    hparams = merge_args_cfg(mock_args, cfg)
    
    # Get denormalization params
    denorm_params = getattr(hparams, 'vis_denorm_params', None)
    
    # Get skeleton format
    skeleton_format = args.skeleton_format
    if skeleton_format is None:
        skeleton_format = getattr(hparams, 'vis_skl_format', 'smpl')
    skeleton_class = get_skeleton_class(skeleton_format)
    print(f"Using skeleton format: {skeleton_format}")
    
    dataset_cfg, pipeline_cfg = _resolve_dataset_cfg(hparams, args.split)
    
    # Create dataset
    print(f"Creating {args.split} dataset: {dataset_cfg['name']}")
    dataset, collate_fn = create_dataset(
        dataset_cfg['name'],
        dataset_cfg['params'],
        pipeline_cfg
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Validate indices
    if args.sample_idx is not None:
        if args.sample_idx < 0 or args.sample_idx >= len(dataset):
            print(f"Error: sample_idx ({args.sample_idx}) out of range [0, {len(dataset)-1}]")
            return
        sample_indices = [args.sample_idx]
    else:
        end_idx = min(args.start_idx + args.num_samples, len(dataset))
        if args.start_idx >= len(dataset):
            print(f"Error: start_idx ({args.start_idx}) >= dataset size ({len(dataset)})")
            return
        sample_indices = list(range(args.start_idx, end_idx))
    
    # Load model
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = LitModel(hparams=hparams)
    
    # Load checkpoint weights
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(args.device)
    model.eval()
    print("Model loaded successfully")
    
    # Resolve SMPL model path if not provided
    if args.smpl_model is None:
        smpl_model_path = getattr(hparams, "smpl_model_path", None)
        if smpl_model_path is None and hasattr(hparams, "smpl_head"):
            smpl_params = hparams.smpl_head.get("params", {})
            smpl_model_path = smpl_params.get("smpl_model_path", smpl_params.get("smpl_path"))
        if smpl_model_path is None:
            smpl_model_path = "weights/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
        args.smpl_model = smpl_model_path

    # Load SMPL model if needed
    smpl_model = None
    has_smpl = model.with_smpl_head and not args.no_mesh
    if has_smpl:
        print(f"Loading SMPL model from: {args.smpl_model}")
        smpl_model = load_smpl_model(args.smpl_model, args.device)
        print("SMPL model loaded")
    
    # Initialize Rerun
    print(f"Initializing Rerun with recording: {args.recording_name}")
    init_rerun_session(
        recording_name=args.recording_name,
        save_rrd=args.save_rrd,
        no_serve=args.no_serve,
        web_port=args.web_port,
        grpc_port=args.grpc_port,
    )
    if not args.no_serve:
        print(f"\n{'='*70}")
        print(f"Rerun visualization server is running!")
        print(f"{'='*70}")
        print(f"Web viewer:    http://localhost:{args.web_port}")
        print(f"Desktop app:   rerun --connect rerun+http://localhost:{args.grpc_port}")
        if args.save_rrd:
            print(f"Recording:     {args.save_rrd}")
        print(f"{'='*70}\n")
    elif args.save_rrd:
        print("Live serving disabled (--no_serve). Only saving to file.")
        print(f"{'='*70}\n")
    
    # Set up blueprint from config-driven modality/view layout
    dataset_params = dataset_cfg.get("params", {})
    input_layout = build_input_layout_from_config(dataset_params)
    depth_as_pointcloud = bool(
        getattr(dataset, "convert_depth_to_lidar", dataset_params.get("convert_depth_to_lidar", False))
    )
    if depth_as_pointcloud:
        for item in input_layout:
            if item.get("modality") == "depth":
                item["view_type"] = "3d"
    blueprint = create_rerun_blueprint(input_layout)
    rr.send_blueprint(blueprint)
    expected_views = {item["modality"]: item["num_views"] for item in input_layout}
    
    # Log static coordinate system
    init_world_axes(
        rr.ViewCoordinates.RIGHT_HAND_Y_DOWN if sensor_mode else rr.ViewCoordinates.RIGHT_HAND_Y_UP
    )
    
    # Process samples
    cross_sample_mode = False
    if args.sample_idx is not None:
        anchor_idx = sample_indices[0]
        anchor_temporal_len = _infer_temporal_length(dataset[anchor_idx])
        if anchor_temporal_len <= 1 and args.num_frames > 1:
            cross_sample_mode = True
            steps = select_sample_frame_steps(
                dataset_size=len(dataset),
                sample_idx=anchor_idx,
                temporal_len=anchor_temporal_len,
                frame_index=args.frame_index,
                num_frames=args.num_frames,
            )
            sample_indices = [sidx for sidx, _ in steps]
            print(
                f"\nProcessing cross-sample timeline from sample {sample_indices[0]} "
                f"to {sample_indices[-1]} ({len(sample_indices)} steps)..."
            )
        else:
            print(f"\nProcessing single sample index {args.sample_idx}...")
    else:
        print(f"\nProcessing samples {sample_indices[0]} to {sample_indices[-1]}...")

    timeline_step = 0
    for idx in tqdm(sample_indices, desc="Visualizing"):
        # Get sample
        sample = dataset[idx]
        
        # Collate into batch
        batch = collate_fn([sample])
        
        sample_id = None
        if 'sample_id' in sample:
            sample_id = sample['sample_id']
            if isinstance(sample_id, (list, tuple)):
                sample_id = sample_id[0]
        reference_extrinsic = None
        image_hw = None
        temporal_len = _infer_temporal_length(sample)
        if cross_sample_mode:
            frame_indices = [0]
        else:
            frame_indices = select_frame_indices(temporal_len, args.frame_index, args.num_frames)
        if sensor_mode:
            reference_extrinsic = resolve_reference_extrinsic(
                sample,
                sensor_label=args.reference_sensor,
                view_index=int(args.reference_view),
            )
        
        # Run inference
        try:
            pred_dict = run_inference(model, batch, args.device)
        except Exception as e:
            print(f"Inference failed for sample {idx}: {e}")
            continue

        pred_extrinsic = None
        if sensor_mode:
            if image_hw is None:
                image_hw = _infer_image_hw(sample)
            pred_extrinsic = _resolve_predicted_extrinsic(
                pred_dict,
                sample,
                sensor_label=args.reference_sensor,
                image_hw=image_hw,
            )
            if pred_extrinsic is None:
                rr.log(
                    "world/info/pred_camera_status",
                    rr.TextLog("pred_camera_unavailable_fallback_to_gt_reference"),
                )
            else:
                rr.log("world/info/pred_camera_status", rr.TextLog("pred_camera_used"))
        for local_t, source_frame_idx in enumerate(frame_indices):
            set_frame_timeline(timeline_step, sample_id=str(sample_id) if sample_id is not None else None)
            timeline_step += 1
            rr.log('world/info/coord_space', rr.TextLog(args.coord_space))
            rr.log('world/info/reference_sensor', rr.TextLog(str(args.reference_sensor)))
            rr.log('world/info/reference_view', rr.TextLog(str(args.reference_view)))
            rr.log('world/info/source_frame_index', rr.TextLog(str(source_frame_idx)))

            # Log configured modality views
            if 'input_rgb' in sample:
                expected_rgb_views = _expected_views_for_sample(sample, "rgb", expected_views.get("rgb"))
                for v, rgb_view in enumerate(
                    split_sample_views(sample["input_rgb"], expected_views=expected_rgb_views, modality="rgb")
                ):
                    rgb_data = process_image_for_display(rgb_view, denorm_params, 'rgb', keep_temporal=True)
                    rgb_image = rgb_data if rgb_data.ndim == 3 else rgb_data[source_frame_idx]
                    if image_hw is None and isinstance(rgb_image, np.ndarray) and rgb_image.ndim >= 2:
                        image_hw = (int(rgb_image.shape[0]), int(rgb_image.shape[1]))
                    rr.log(f"world/inputs/rgb/view_{v}/image", rr.Image(rgb_image))

            if 'input_depth' in sample or (depth_as_pointcloud and 'input_lidar' in sample):
                expected_depth_views = _expected_views_for_sample(sample, "depth", expected_views.get("depth"))
                if depth_as_pointcloud and 'input_lidar' in sample:
                    for v, depth_pc_view in enumerate(
                        split_sample_views(
                            sample["input_lidar"],
                            expected_views=expected_depth_views,
                            modality="depth",
                        )
                    ):
                        depth_pc = depth_pc_view
                        if isinstance(depth_pc, torch.Tensor):
                            depth_pc = depth_pc.cpu().numpy()
                        if depth_pc.ndim == 3:
                            depth_pc = depth_pc[source_frame_idx]
                        if not sensor_mode:
                            depth_pc = rotate_points_180_y(depth_pc)
                        log_point_cloud_3d(
                            f"world/inputs/depth/view_{v}",
                            depth_pc,
                            colors=[120, 180, 255],
                            radii=0.02,
                        )
                else:
                    for v, depth_view in enumerate(
                        split_sample_views(
                            sample["input_depth"],
                            expected_views=expected_depth_views,
                            modality="depth",
                        )
                    ):
                        depth_data = process_image_for_display(depth_view, denorm_params, 'depth', keep_temporal=True)
                        depth_image = depth_data if depth_data.ndim == 2 else depth_data[source_frame_idx]
                        depth_valid = depth_image[depth_image > 0]
                        if len(depth_valid) > 0:
                            vmin, vmax = np.percentile(depth_valid, [2, 98])
                            depth_norm = np.clip(depth_image, vmin, vmax)
                            depth_norm = ((depth_norm - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)
                        else:
                            depth_norm = depth_image.astype(np.uint8)
                        rr.log(f"world/inputs/depth/view_{v}/image", rr.DepthImage(depth_norm))

            if 'input_lidar' in sample and not depth_as_pointcloud:
                expected_lidar_views = _expected_views_for_sample(sample, "lidar", expected_views.get("lidar"))
                for v, lidar_view in enumerate(
                    split_sample_views(
                        sample["input_lidar"],
                        expected_views=expected_lidar_views,
                        modality="lidar",
                    )
                ):
                    lidar_data = lidar_view
                    if isinstance(lidar_data, torch.Tensor):
                        lidar_data = lidar_data.cpu().numpy()
                    if lidar_data.ndim == 3:
                        lidar_data = lidar_data[source_frame_idx]
                    if not sensor_mode:
                        lidar_data = rotate_points_180_y(lidar_data)
                    log_point_cloud_3d(
                        f"world/inputs/lidar/view_{v}",
                        lidar_data,
                        colors=[100, 200, 100],
                        radii=0.02,
                    )

            if 'input_mmwave' in sample:
                expected_mmwave_views = _expected_views_for_sample(sample, "mmwave", expected_views.get("mmwave"))
                for v, mmwave_view in enumerate(
                    split_sample_views(
                        sample["input_mmwave"],
                        expected_views=expected_mmwave_views,
                        modality="mmwave",
                    )
                ):
                    mmwave_data = mmwave_view
                    if isinstance(mmwave_data, torch.Tensor):
                        mmwave_data = mmwave_data.cpu().numpy()
                    if mmwave_data.ndim == 3:
                        mmwave_data = mmwave_data[source_frame_idx]
                    if not sensor_mode:
                        mmwave_data = rotate_points_180_y(mmwave_data)
                    log_point_cloud_3d(
                        f"world/inputs/mmwave/view_{v}",
                        mmwave_data,
                        colors=[255, 150, 50],
                        radii=0.03,
                    )

            # Process SMPL predictions
            if 'pred_smpl' in pred_dict and has_smpl:
                pred_smpl = pred_dict['pred_smpl']
                if 'pred_smpl_params' in pred_smpl:
                    pred_vec = torch2numpy(pred_smpl['pred_smpl_params'])
                    if np.asarray(pred_vec).ndim >= 2:
                        pred_vec = _take_temporal(pred_vec, source_frame_idx)
                    smpl_params = split_smpl_params(pred_vec)
                else:
                    smpl_params = {
                        'global_orient': torch2numpy(pred_smpl['global_orient']),
                        'body_pose': torch2numpy(pred_smpl['body_pose']),
                        'betas': torch2numpy(pred_smpl['betas']),
                        'transl': torch2numpy(pred_smpl['transl']),
                    }
                if args.canonical:
                    smpl_params = canonicalize_smpl_params(smpl_params)
                vertices, joints, faces = smpl_params_to_mesh(
                    smpl_model, smpl_params, args.device
                )
                if sensor_mode:
                    pred_ref_extrinsic = pred_extrinsic if pred_extrinsic is not None else reference_extrinsic
                    vertices = transform_points_to_camera(vertices, pred_ref_extrinsic)
                    joints = transform_points_to_camera(joints, pred_ref_extrinsic)
                else:
                    if args.canonical and args.canonical_upright:
                        vertices = rotate_points_x(vertices, np.deg2rad(180.0))
                        joints = rotate_points_x(joints, np.deg2rad(180.0))
                    vertices = rotate_points_180_y(vertices)
                    joints = rotate_points_180_y(joints)

                log_smpl_views(
                    "prediction",
                    vertices[0],
                    faces,
                    joints[0],
                    skeleton_class,
                    mesh_color=(100, 180, 255),
                    mesh_alpha=0.3,
                    skeleton_color=(0, 200, 255),
                    skeleton_radius=0.02,
                )
            elif 'pred_keypoints' in pred_dict:
                pred_kpts = _take_temporal(pred_dict['pred_keypoints'], source_frame_idx)
                if pred_kpts.ndim == 3:
                    pred_kpts = pred_kpts[0]
                if sensor_mode:
                    pred_ref_extrinsic = pred_extrinsic if pred_extrinsic is not None else reference_extrinsic
                    pred_kpts = transform_points_to_camera(pred_kpts, pred_ref_extrinsic)
                else:
                    if args.canonical:
                        pred_kpts = pred_kpts - pred_kpts[0:1]
                        if args.canonical_upright:
                            pred_kpts = rotate_points_x(pred_kpts, np.deg2rad(180.0))
                    pred_kpts = rotate_points_180_y(pred_kpts)

                log_skeleton_views(
                    "prediction",
                    pred_kpts,
                    skeleton_class,
                    color=(0, 200, 255),
                    radius=0.02,
                )

            # Log ground truth
            if args.show_gt:
                gt_vertices = None
                gt_joints = None
                faces = None
                if has_smpl:
                    gt_smpl_params = None
                    if 'gt_smpl' in sample:
                        gt_smpl = sample['gt_smpl']
                        gt_smpl_params = {}
                        for key in ['global_orient', 'body_pose', 'betas', 'transl']:
                            if key in gt_smpl:
                                val = gt_smpl[key]
                                if isinstance(val, torch.Tensor):
                                    val = val.cpu().numpy()
                                gt_smpl_params[key] = val
                    elif 'gt_smpl_params_seq' in sample or 'gt_smpl_params' in sample:
                        gt_vec = sample.get('gt_smpl_params_seq', sample.get('gt_smpl_params'))
                        if isinstance(gt_vec, torch.Tensor):
                            gt_vec = gt_vec.cpu().numpy()
                        if np.asarray(gt_vec).ndim >= 2:
                            gt_vec = _take_temporal(gt_vec, source_frame_idx)
                        gt_smpl_params = split_smpl_params(gt_vec)

                    if gt_smpl_params is not None and len(gt_smpl_params) == 4:
                        if args.canonical:
                            gt_smpl_params = canonicalize_smpl_params(gt_smpl_params)

                        gt_vertices, gt_joints, faces = smpl_params_to_mesh(
                            smpl_model, gt_smpl_params, args.device
                        )
                        if sensor_mode:
                            gt_vertices = transform_points_to_camera(gt_vertices, reference_extrinsic)
                            gt_joints = transform_points_to_camera(gt_joints, reference_extrinsic)
                        else:
                            if args.canonical and args.canonical_upright:
                                gt_vertices = rotate_points_x(gt_vertices, np.deg2rad(180.0))
                                gt_joints = rotate_points_x(gt_joints, np.deg2rad(180.0))
                            gt_vertices = rotate_points_180_y(gt_vertices)
                            gt_joints = rotate_points_180_y(gt_joints)

                if 'gt_keypoints_seq' in sample or 'gt_keypoints' in sample:
                    gt_kpts = sample.get('gt_keypoints_seq', sample.get('gt_keypoints'))
                    gt_kpts = _take_temporal(gt_kpts, source_frame_idx)
                    if gt_kpts.ndim == 3:
                        gt_kpts = gt_kpts[0]
                    if sensor_mode:
                        gt_kpts = transform_points_to_camera(gt_kpts, reference_extrinsic)
                    else:
                        if args.canonical:
                            gt_kpts = gt_kpts - gt_kpts[0:1]
                            if args.canonical_upright:
                                gt_kpts = rotate_points_x(gt_kpts, np.deg2rad(180.0))
                        gt_kpts = rotate_points_180_y(gt_kpts)
                    if gt_joints is not None:
                        # Align pelvis to reduce vertical offset between GT kpts and SMPL joints.
                        gt_joints_align = gt_joints[0] if gt_joints.ndim == 3 else gt_joints
                        gt_kpts = gt_kpts - (gt_kpts[0] - gt_joints_align[0])
                        gt_kpts = align_keypoints_to_joints_scale(gt_kpts, gt_joints_align)

                    log_skeleton_views(
                        "ground_truth",
                        gt_kpts,
                        skeleton_class,
                        color=(0, 255, 100),
                        radius=0.02,
                    )

                if gt_vertices is not None and gt_joints is not None and faces is not None:
                    gt_mesh_vertices = gt_vertices[0] if gt_vertices.ndim == 3 else gt_vertices
                    log_smpl_views(
                        "ground_truth",
                        gt_mesh_vertices,
                        faces,
                        gt_joints[0],
                        skeleton_class,
                        mesh_color=(100, 255, 150),
                        mesh_alpha=0.3,
                        skeleton_color=(0, 255, 100),
                        skeleton_radius=0.02,
                        log_skeleton=True,
                    )
    rr.log("world/info/num_visualized_frames", rr.TextLog(str(timeline_step)), static=True)

    print("\nVisualization complete!")
    if args.save_rrd:
        print(f"Recording saved to: {args.save_rrd}")
    
    if not args.no_serve:
        print(f"\nWeb viewer is still running at: http://localhost:{args.web_port}")
        print("Use the Rerun viewer to explore the results:")
        print("  - Use timeline slider to navigate through frames")
        print("  - Click on entities in the left panel to show/hide")
        print("  - Use mouse to rotate/zoom the 3D view")
        
        # Keep the web server running
        print("\nPress Ctrl+C to exit...")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting...")
    else:
        print("\nTo view the recording, run:")
        print(f"  rerun {args.save_rrd}")


if __name__ == '__main__':
    main()
