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
from misc.registry import create_dataset
from models.model_api import LitModel
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
            pred_dict['pred_cameras'] = model.camera_head.predict(feats_agg)
    
    return pred_dict


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
    
    args = parser.parse_args()
    
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
    
    # Select dataset config based on split
    if args.split == 'train':
        dataset_cfg = hparams.train_dataset
        pipeline_cfg = hparams.train_pipeline
    elif args.split == 'val':
        dataset_cfg = hparams.val_dataset
        pipeline_cfg = hparams.val_pipeline
    else:
        dataset_cfg = hparams.test_dataset
        pipeline_cfg = hparams.test_pipeline
    
    # Create dataset
    print(f"Creating {args.split} dataset: {dataset_cfg['name']}")
    dataset, collate_fn = create_dataset(
        dataset_cfg['name'],
        dataset_cfg['params'],
        pipeline_cfg
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Validate indices
    end_idx = min(args.start_idx + args.num_samples, len(dataset))
    if args.start_idx >= len(dataset):
        print(f"Error: start_idx ({args.start_idx}) >= dataset size ({len(dataset)})")
        return
    
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
    blueprint = create_rerun_blueprint(input_layout)
    rr.send_blueprint(blueprint)
    expected_views = {item["modality"]: item["num_views"] for item in input_layout}
    
    # Log static coordinate system
    init_world_axes()
    
    # Process samples
    print(f"\nProcessing samples {args.start_idx} to {end_idx - 1}...")
    
    for idx in tqdm(range(args.start_idx, end_idx), desc="Visualizing"):
        # Get sample
        sample = dataset[idx]
        
        # Collate into batch
        batch = collate_fn([sample])
        
        sample_id = None
        if 'sample_id' in sample:
            sample_id = sample['sample_id']
            if isinstance(sample_id, (list, tuple)):
                sample_id = sample_id[0]
        set_frame_timeline(idx, sample_id=str(sample_id) if sample_id is not None else None)
        
        # Log configured modality views
        if 'input_rgb' in sample:
            for v, rgb_view in enumerate(
                split_sample_views(sample["input_rgb"], expected_views=expected_views.get("rgb"), modality="rgb")
            ):
                rgb_image = process_image_for_display(rgb_view, denorm_params, 'rgb')
                rr.log(f"world/inputs/rgb/view_{v}/image", rr.Image(rgb_image))

        if 'input_depth' in sample:
            for v, depth_view in enumerate(
                split_sample_views(
                    sample["input_depth"],
                    expected_views=expected_views.get("depth"),
                    modality="depth",
                )
            ):
                depth_image = process_image_for_display(depth_view, denorm_params, 'depth')
                depth_valid = depth_image[depth_image > 0]
                if len(depth_valid) > 0:
                    vmin, vmax = np.percentile(depth_valid, [2, 98])
                    depth_norm = np.clip(depth_image, vmin, vmax)
                    depth_norm = ((depth_norm - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)
                else:
                    depth_norm = depth_image.astype(np.uint8)
                rr.log(f"world/inputs/depth/view_{v}/image", rr.DepthImage(depth_norm))

        if 'input_lidar' in sample:
            for v, lidar_view in enumerate(
                split_sample_views(
                    sample["input_lidar"],
                    expected_views=expected_views.get("lidar"),
                    modality="lidar",
                )
            ):
                lidar_data = lidar_view
                if isinstance(lidar_data, torch.Tensor):
                    lidar_data = lidar_data.cpu().numpy()
                if lidar_data.ndim == 3:
                    lidar_data = lidar_data[-1]
                lidar_data = rotate_points_180_y(lidar_data)
                log_point_cloud_3d(
                    f"world/inputs/lidar/view_{v}",
                    lidar_data,
                    colors=[100, 200, 100],
                    radii=0.02,
                )

        if 'input_mmwave' in sample:
            for v, mmwave_view in enumerate(
                split_sample_views(
                    sample["input_mmwave"],
                    expected_views=expected_views.get("mmwave"),
                    modality="mmwave",
                )
            ):
                mmwave_data = mmwave_view
                if isinstance(mmwave_data, torch.Tensor):
                    mmwave_data = mmwave_data.cpu().numpy()
                if mmwave_data.ndim == 3:
                    mmwave_data = mmwave_data[-1]
                mmwave_data = rotate_points_180_y(mmwave_data)
                log_point_cloud_3d(
                    f"world/inputs/mmwave/view_{v}",
                    mmwave_data,
                    colors=[255, 150, 50],
                    radii=0.03,
                )
        
        # Run inference
        try:
            pred_dict = run_inference(model, batch, args.device)
        except Exception as e:
            print(f"Inference failed for sample {idx}: {e}")
            continue
        
        # Process SMPL predictions
        if 'pred_smpl' in pred_dict and has_smpl:
            pred_smpl = pred_dict['pred_smpl']
            
            # Convert tensors to numpy
            if 'pred_smpl_params' in pred_smpl:
                smpl_params = split_smpl_params(torch2numpy(pred_smpl['pred_smpl_params']))
            else:
                smpl_params = {
                    'global_orient': torch2numpy(pred_smpl['global_orient']),
                    'body_pose': torch2numpy(pred_smpl['body_pose']),
                    'betas': torch2numpy(pred_smpl['betas']),
                    'transl': torch2numpy(pred_smpl['transl']),
                }
            if args.canonical:
                smpl_params = canonicalize_smpl_params(smpl_params)
            
            # Get mesh and joints
            vertices, joints, faces = smpl_params_to_mesh(
                smpl_model, smpl_params, args.device
            )
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
        
        # Process keypoint predictions (if available and no SMPL)
        elif 'pred_keypoints' in pred_dict:
            pred_kpts = torch2numpy(pred_dict['pred_keypoints'])
            if pred_kpts.ndim == 3:
                pred_kpts = pred_kpts[0]  # First batch
            if pred_kpts.ndim == 3:
                pred_kpts = pred_kpts[-1]  # Last temporal frame
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
            # GT SMPL mesh
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
                elif 'gt_smpl_params' in sample:
                    gt_vec = sample['gt_smpl_params']
                    if isinstance(gt_vec, torch.Tensor):
                        gt_vec = gt_vec.cpu().numpy()
                    gt_smpl_params = split_smpl_params(gt_vec)

                gt_vertices = None
                gt_joints = None
                if gt_smpl_params is not None and len(gt_smpl_params) == 4:
                    if args.canonical:
                        gt_smpl_params = canonicalize_smpl_params(gt_smpl_params)

                    gt_vertices, gt_joints, faces = smpl_params_to_mesh(
                        smpl_model, gt_smpl_params, args.device
                    )
                    if args.canonical and args.canonical_upright:
                        gt_vertices = rotate_points_x(gt_vertices, np.deg2rad(180.0))
                        gt_joints = rotate_points_x(gt_joints, np.deg2rad(180.0))
                    gt_vertices = rotate_points_180_y(gt_vertices)
                    gt_joints = rotate_points_180_y(gt_joints)

                # GT keypoints
                if 'gt_keypoints' in sample:
                    gt_kpts = sample['gt_keypoints']
                    if isinstance(gt_kpts, torch.Tensor):
                        gt_kpts = gt_kpts.cpu().numpy()

                    # Handle temporal dimension
                    if gt_kpts.ndim == 3:
                        gt_kpts = gt_kpts[-1]  # Last frame
                    if args.canonical:
                        gt_kpts = gt_kpts - gt_kpts[0:1]
                        if args.canonical_upright:
                            gt_kpts = rotate_points_x(gt_kpts, np.deg2rad(180.0))
                    gt_kpts = rotate_points_180_y(gt_kpts)
                    if gt_joints is not None:
                        # Align pelvis to reduce vertical offset between GT kpts and SMPL joints.
                        gt_kpts = gt_kpts - (gt_kpts[0] - gt_joints[0])
                        gt_kpts = align_keypoints_to_joints_scale(gt_kpts, gt_joints)

                    log_skeleton_views(
                        "ground_truth",
                        gt_kpts,
                        skeleton_class,
                        color=(0, 255, 100),
                        radius=0.02,
                    )

                if gt_vertices is not None and gt_joints is not None:
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
                        log_skeleton=False,
                    )
    
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
