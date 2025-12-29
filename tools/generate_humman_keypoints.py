"""
Generate 3D keypoints from SMPL parameters for Humman dataset.

This script processes all sequences in the Humman dataset and generates 3D keypoints
from SMPL parameters using the SMPL model. The keypoints are saved as keypoints_3d.npz
in each sequence directory.

Usage:
    python tools/generate_humman_keypoints.py --data_root /data/shared/humman_release_v1.0_point \
                                               --smpl_model_path /path/to/smpl/models
"""

import os
import os.path as osp
import numpy as np
import argparse
import glob
from tqdm import tqdm
import torch

# Add parent directory to path to import SMPL
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from models.smpl.smpl import SMPL


def convert_smpl_to_keypoints(smpl_model, global_orient, body_pose, betas, transl, device='cuda'):
    """
    Convert SMPL parameters to 3D keypoints.
    
    Args:
        smpl_model: SMPL model instance
        global_orient: (N, 3) array
        body_pose: (N, 69) array  
        betas: (N, 10) array
        transl: (N, 3) array
        device: 'cuda' or 'cpu'
    
    Returns:
        keypoints_3d: (N, 24, 3) array of 3D joint positions
    """
    N = global_orient.shape[0]
    
    # Concatenate global_orient and body_pose to get full pose
    # SMPL expects (N, 72) = (N, 24*3) pose parameters
    pose = np.concatenate([global_orient, body_pose], axis=1)  # (N, 72)
    
    # Get the expected number of beta parameters from the model
    # The model's th_betas has shape (1, num_betas), typically 300
    expected_betas = smpl_model.th_betas.shape[1]
    
    # Pad betas if necessary (dataset has 10 betas, model may expect 300)
    if betas.shape[1] < expected_betas:
        betas_padded = np.zeros((N, expected_betas), dtype=betas.dtype)
        betas_padded[:, :betas.shape[1]] = betas
        betas = betas_padded
    
    # Convert to torch tensors
    th_pose = torch.from_numpy(pose).float().to(device)
    th_betas = torch.from_numpy(betas).float().to(device)
    th_trans = torch.from_numpy(transl).float().to(device)
    
    # Forward pass through SMPL model
    with torch.no_grad():
        output = smpl_model(th_pose, th_betas, th_trans)
        # Output is typically (vertices, joints) or just joints
        # The joints should be the second element or the only element
        if isinstance(output, (list, tuple)):
            joints = output[1]  # Typically joints are the second output
        else:
            joints = output
    
    # Convert back to numpy
    keypoints_3d = joints.cpu().numpy()  # (N, 24, 3)
    
    return keypoints_3d


def process_sequence(seq_dir, smpl_model, device='cuda', force=False):
    """
    Process a single sequence: load SMPL params, generate keypoints, save to file.
    
    Args:
        seq_dir: Path to sequence directory
        smpl_model: SMPL model instance
        device: 'cuda' or 'cpu'
        force: If True, regenerate even if keypoints_3d.npz already exists
    
    Returns:
        True if successful, False otherwise
    """
    seq_name = osp.basename(seq_dir)
    keypoints_file = osp.join(seq_dir, "keypoints_3d.npz")
    
    # Skip if already processed (unless force=True)
    if osp.exists(keypoints_file) and not force:
        return True
    
    # Check if SMPL params exist
    smpl_file = osp.join(seq_dir, "smpl_params.npz")
    if not osp.exists(smpl_file):
        print(f"Warning: {seq_name} - smpl_params.npz not found, skipping")
        return False
    
    try:
        # Load SMPL parameters
        smpl_data = np.load(smpl_file)
        global_orient = smpl_data['global_orient']  # (N, 3)
        body_pose = smpl_data['body_pose']  # (N, 69)
        betas = smpl_data['betas']  # (N, 10)
        transl = smpl_data['transl']  # (N, 3)
        
        num_frames = global_orient.shape[0]
        
        # Convert SMPL to keypoints
        keypoints_3d = convert_smpl_to_keypoints(
            smpl_model, global_orient, body_pose, betas, transl, device
        )
        
        # Save keypoints
        np.savez_compressed(
            keypoints_file,
            keypoints_3d=keypoints_3d,  # (N, 24, 3)
            num_frames=num_frames
        )
        
        return True
        
    except Exception as e:
        print(f"Error processing {seq_name}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate 3D keypoints from SMPL parameters for Humman dataset")
    parser.add_argument('--data_root', type=str, default='/data/shared/humman_release_v1.0_point',
                        help='Path to Humman dataset root directory')
    parser.add_argument('--smpl_model_path', type=str, default='weights/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl',
                        help='Path to SMPL model file')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device for computation')
    parser.add_argument('--force', action='store_true',
                        help='Force regeneration even if keypoints already exist')
    args = parser.parse_args()

    smpl_model_file = args.smpl_model_path
    
    # Check if SMPL model file exists
    if not osp.exists(smpl_model_file):
        print(f"Error: SMPL model file not found: {smpl_model_file}")
        print("\nPlease download SMPL models from: https://smpl.is.tue.mpg.de/")
        print("And extract them to the specified path.")
        print(f"\nExpected file: {smpl_model_file}")
        return
    
    # Initialize SMPL model
    print(f"Loading SMPL model ({args.gender}) from {smpl_model_file}...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    try:
        smpl_model = SMPL(
            model_path=smpl_model_file,
            gender=args.gender
        ).to(device)
        smpl_model.eval()
        print(f"SMPL model loaded successfully on {device}")
        print(f"Number of joints: {smpl_model.num_joints}")
    except Exception as e:
        print(f"Error loading SMPL model: {str(e)}")
        print("\nMake sure you have the correct SMPL model files:")
        print(f"  - {smpl_model_file}")
        return
    
    # Get all sequence directories
    print(f"\nScanning dataset at {args.data_root}...")
    seq_dirs = glob.glob(osp.join(args.data_root, "p*_a*"))
    seq_dirs = sorted([d for d in seq_dirs if osp.isdir(d)])
    
    print(f"Found {len(seq_dirs)} sequences")
    
    if len(seq_dirs) == 0:
        print("No sequences found! Check the data_root path.")
        return
    
    # Process each sequence
    success_count = 0
    skip_count = 0
    error_count = 0
    
    print("\nProcessing sequences...")
    for seq_dir in tqdm(seq_dirs, desc="Generating keypoints"):
        keypoints_file = osp.join(seq_dir, "keypoints_3d.npz")
        
        if osp.exists(keypoints_file) and not args.force:
            skip_count += 1
            continue
        
        if process_sequence(seq_dir, smpl_model, device, args.force):
            success_count += 1
        else:
            error_count += 1
    
    # Print summary
    print("\n" + "="*60)
    print("Processing complete!")
    print(f"  Successfully processed: {success_count}")
    print(f"  Skipped (already exists): {skip_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total sequences: {len(seq_dirs)}")
    print("="*60)
    
    if error_count > 0:
        print("\nSome sequences failed to process. Check the warnings above.")


if __name__ == '__main__':
    main()
