import os
import os.path as osp
import cv2
import numpy as np
import json
import glob
import random
import re
from typing import Callable, List, Optional, Sequence, Tuple, Union

from datasets.base_dataset import BaseDataset
import warnings


def axis_angle_to_matrix_np(axis_angle: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = axis_angle / angle
    x, y, z = axis
    K = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float32,
    )
    eye = np.eye(3, dtype=np.float32)
    return eye + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def matrix_to_axis_angle_np(rot: np.ndarray) -> np.ndarray:
    trace = np.trace(rot)
    cos = (trace - 1.0) / 2.0
    cos = np.clip(cos, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float32)
    rx = rot[2, 1] - rot[1, 2]
    ry = rot[0, 2] - rot[2, 0]
    rz = rot[1, 0] - rot[0, 1]
    rvec = np.array([rx, ry, rz], dtype=np.float32)
    denom = 2.0 * np.sin(angle)
    if abs(denom) < 1e-6:
        return 0.5 * rvec
    axis = rvec / denom
    return axis * angle


class HummanDataset(BaseDataset):
    def __init__(
        self,
        data_root: str = "/data/shared/humman_release_v1.0_point",
        unit: str = "m",
        pipeline: List[dict] = [],
        split: str = "train",
        modality_names: Sequence[str] = ["rgb", "depth"],
        rgb_cameras: Optional[Sequence[str]] = None,  # None means all cameras
        depth_cameras: Optional[Sequence[str]] = None,  # None means all cameras
        anchor_key: str = "input_rgb",
        seq_len: int = 5,
        seq_step: int = 1,
        pad_seq: bool = False,
        causal: bool = False,
        use_all_pairs: bool = False,  # False: random pair per frame, True: all pairs
        colocated: bool = False, # If true, only use colocated RGB-Depth pairs
        random_seed: Optional[int] = None,
    ):
        super().__init__(pipeline=pipeline)
        self.data_root = data_root
        self.split = split
        self.unit = unit
        self.seq_len = seq_len
        self.seq_step = seq_step
        self.causal = causal
        self.pad_seq = pad_seq
        self.modality_names = modality_names
        self.use_all_pairs = use_all_pairs
        self.colocated = colocated
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Available cameras
        self.available_kinect_cameras = [f"kinect_{i:03d}" for i in range(10)]
        self.available_iphone_cameras = ["iphone"]
        
        # Set RGB cameras
        if rgb_cameras is None:
            self.rgb_cameras = self.available_kinect_cameras + self.available_iphone_cameras
        else:
            self.rgb_cameras = list(rgb_cameras)
            
        # Set Depth cameras
        if depth_cameras is None:
            self.depth_cameras = self.available_kinect_cameras + self.available_iphone_cameras
        else:
            self.depth_cameras = list(depth_cameras)
        
        # Validate modality names
        valid_modalities = {"rgb", "depth"}
        invalid_modalities = set(modality_names) - valid_modalities
        if invalid_modalities:
            warnings.warn(
                f"Invalid modality names detected: {invalid_modalities}. "
                f"Only 'rgb' and 'depth' are supported for Humman dataset."
            )
            
        # Validate anchor_key
        valid_anchor_keys = {f"input_{mod}" for mod in modality_names}
        if anchor_key not in valid_anchor_keys:
            warnings.warn(
                f"Invalid anchor_key: {anchor_key}. "
                f"Must be one of {valid_anchor_keys}. "
                f"Defaulting to 'input_rgb'."
            )
            self.anchor_key = "input_rgb" if "rgb" in modality_names else f"input_{modality_names[0]}"
        else:
            self.anchor_key = anchor_key
            
        # Validate unit
        if unit not in {"mm", "m"}:
            warnings.warn(
                f"Invalid unit: {unit}. Defaulting to 'm'."
            )
            self.unit = "m"

        # Build dataset index
        self.data_list = self._build_dataset()

    def _build_dataset(self):
        """Build dataset index with available sequences."""
        data_list = []
        
        # Get all sequence directories
        seq_dirs = glob.glob(osp.join(self.data_root, "p*_a*"))
        seq_dirs = sorted(seq_dirs)
        
        # Split into train/test (80/20 split based on person ID)
        person_ids = sorted(list(set([osp.basename(d).split('_')[0] for d in seq_dirs])))
        split_idx = int(len(person_ids) * 0.8)
        
        if self.split == "train":
            valid_persons = set(person_ids[:split_idx])
        elif self.split == "test":
            valid_persons = set(person_ids[split_idx:])
        elif self.split == "train_mini":
            valid_persons = set(person_ids[:16]) # first 16 persons for mini train
        elif self.split == "test_mini":
            valid_persons = set(person_ids[split_idx:split_idx+4]) # 4 persons for mini test
        else:
            valid_persons = set(person_ids)
        
        for seq_dir in seq_dirs:
            seq_name = osp.basename(seq_dir)
            person_id = seq_name.split('_')[0]
            
            # Skip if not in current split
            if person_id not in valid_persons:
                continue
            
            # Check if required data exists
            if not osp.exists(osp.join(seq_dir, "cameras.json")):
                continue
            if not osp.exists(osp.join(seq_dir, "smpl_params.npz")):
                continue
                
            # Count frames from one RGB camera to get sequence length
            if "rgb" in self.modality_names and self.rgb_cameras:
                # Check kinect cameras first
                kinect_cams = [c for c in self.rgb_cameras if c.startswith("kinect")]
                if kinect_cams:
                    sample_cam = kinect_cams[0]
                    frame_dir = osp.join(seq_dir, "kinect_color", sample_cam)
                else:
                    # Use iPhone
                    sample_cam = "iphone"
                    frame_dir = osp.join(seq_dir, "iphone_color", sample_cam)
                    
                if not osp.exists(frame_dir):
                    continue
                    
                frame_files = glob.glob(osp.join(frame_dir, "*.png"))
                num_frames = len(frame_files)
            elif "depth" in self.modality_names and self.depth_cameras:
                # Use depth to count frames
                kinect_cams = [c for c in self.depth_cameras if c.startswith("kinect")]
                if kinect_cams:
                    sample_cam = kinect_cams[0]
                    frame_dir = osp.join(seq_dir, "kinect_depth", sample_cam)
                else:
                    sample_cam = "iphone"
                    frame_dir = osp.join(seq_dir, "iphone_depth", sample_cam)
                    
                if not osp.exists(frame_dir):
                    continue
                    
                frame_files = glob.glob(osp.join(frame_dir, "*.png"))
                num_frames = len(frame_files)
            else:
                continue
            
            if num_frames < self.seq_len:
                continue
            
            # Create sequences with overlap
            for start_idx in range(0, num_frames - self.seq_len + 1, self.seq_step):
                if self.use_all_pairs:
                    # Create one entry for each RGB-Depth pair
                    for rgb_cam in self.rgb_cameras:
                        for depth_cam in self.depth_cameras:
                            data_info = {
                                "seq_dir": seq_dir,
                                "seq_name": seq_name,
                                "person_id": person_id,
                                "start_frame": start_idx,
                                "num_frames": num_frames,
                                "rgb_camera": rgb_cam,
                                "depth_camera": depth_cam,
                            }
                            data_list.append(data_info)
                else:
                    # Create one entry per sequence, camera pair will be randomly selected
                    data_info = {
                        "seq_dir": seq_dir,
                        "seq_name": seq_name,
                        "person_id": person_id,
                        "start_frame": start_idx,
                        "num_frames": num_frames,
                        "rgb_camera": None,  # Will be randomly selected in __getitem__
                        "depth_camera": None,  # Will be randomly selected in __getitem__
                    }
                    data_list.append(data_info)
        
        return data_list

    def _load_camera_params(self, seq_dir):
        """Load camera parameters from cameras.json."""
        camera_file = osp.join(seq_dir, "cameras.json")
        with open(camera_file, 'r') as f:
            cameras = json.load(f)
        return cameras

    def _load_smpl_params(self, seq_dir):
        """Load SMPL parameters from smpl_params.npz."""
        smpl_file = osp.join(seq_dir, "smpl_params.npz")
        smpl_data = np.load(smpl_file)
        return {
            'global_orient': smpl_data['global_orient'],  # (num_frames, 3)
            'body_pose': smpl_data['body_pose'],  # (num_frames, 69)
            'betas': smpl_data['betas'],  # (num_frames, 10)
            'transl': smpl_data['transl'],  # (num_frames, 3)
        }
    
    def _load_keypoints_3d(self, seq_dir):
        """Load precomputed 3D keypoints and vertices from keypoints_3d.npz."""
        keypoints_file = osp.join(seq_dir, "keypoints_3d.npz")
        if osp.exists(keypoints_file):
            data = np.load(keypoints_file)
            result = {
                'keypoints_3d': data['keypoints_3d']  # (num_frames, 24, 3)
            }
            # Load vertices if available
            if 'vertices' in data:
                result['vertices'] = data['vertices']  # (num_frames, 6890, 3)
            return result
        else:
            warnings.warn(
                f"keypoints_3d.npz not found in {seq_dir}. "
                f"Run tools/generate_humman_keypoints.py to precompute keypoints."
            )
            return None
            return None

    def _load_rgb_frames(self, seq_dir, camera_name, start_frame, seq_len):
        """Load RGB frame sequence."""
        frames = []
        
        if camera_name.startswith("kinect"):
            base_dir = osp.join(seq_dir, "kinect_color", camera_name)
            # Kinect RGB frames start at 000001.png
            frame_offset = 1
        else:
            base_dir = osp.join(seq_dir, "iphone_color", camera_name)
            # iPhone RGB frames start at 000001.png
            frame_offset = 1
        
        for i in range(seq_len):
            frame_idx = start_frame + i + frame_offset
            frame_path = osp.join(base_dir, f"{frame_idx:06d}.png")
            
            if osp.exists(frame_path):
                frame = cv2.imread(frame_path)
                
                # TODO: This is a temporary fallback for corrupted PNG files (CRC errors).
                # We should fix the dataset by re-downloading/regenerating corrupted files.
                # See scripts/check_humman_dataset.py to identify corrupted files.
                if frame is None:
                    warnings.warn(
                        f"Failed to load RGB frame (corrupted?): {frame_path}, "
                        "using fallback"
                    )
                    if frames:
                        # Use previous frame as fallback
                        frames.append(frames[-1].copy())
                    else:
                        # First frame is corrupted - try to find next valid frame
                        fallback_frame = None
                        for j in range(i + 1, seq_len):
                            next_frame_path = osp.join(base_dir, f"{start_frame + j + frame_offset:06d}.png")
                            if osp.exists(next_frame_path):
                                fallback_frame = cv2.imread(next_frame_path)
                                if fallback_frame is not None:
                                    fallback_frame = cv2.cvtColor(fallback_frame, cv2.COLOR_BGR2RGB)
                                    break
                        if fallback_frame is not None:
                            frames.append(fallback_frame)
                        else:
                            # Create a zero placeholder as last resort (512x512 RGB)
                            frames.append(np.zeros((512, 512, 3), dtype=np.uint8))
                    continue
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # Handle missing frames
                warnings.warn(f"RGB frame not found: {frame_path}, using fallback")
                if frames:
                    # Repeat last frame
                    frames.append(frames[-1].copy())
                else:
                    # First frame is missing - try to find next valid frame
                    fallback_frame = None
                    for j in range(i + 1, seq_len):
                        next_frame_path = osp.join(base_dir, f"{start_frame + j + frame_offset:06d}.png")
                        if osp.exists(next_frame_path):
                            fallback_frame = cv2.imread(next_frame_path)
                            if fallback_frame is not None:
                                fallback_frame = cv2.cvtColor(fallback_frame, cv2.COLOR_BGR2RGB)
                                break
                    if fallback_frame is not None:
                        frames.append(fallback_frame)
                    else:
                        # Create a zero placeholder as last resort (512x512 RGB)
                        frames.append(np.zeros((512, 512, 3), dtype=np.uint8))
        
        return frames

    def _load_depth_frames(self, seq_dir, camera_name, start_frame, seq_len):
        """Load depth frame sequence."""
        frames = []
        
        if camera_name.startswith("kinect"):
            base_dir = osp.join(seq_dir, "kinect_depth", camera_name)
        else:
            base_dir = osp.join(seq_dir, "iphone_depth", camera_name)
        
        # Depth frames start at 000000.png
        for i in range(seq_len):
            frame_idx = start_frame + i
            frame_path = osp.join(base_dir, f"{frame_idx:06d}.png")
            
            if osp.exists(frame_path):
                # Load as 16-bit depth image
                depth = cv2.imread(frame_path, cv2.IMREAD_ANYDEPTH)
                if depth is None:
                    depth = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
                
                # TODO: This is a temporary fallback for corrupted PNG files (CRC errors).
                # We should fix the dataset by re-downloading/regenerating corrupted files.
                # See scripts/check_humman_dataset.py to identify corrupted files.
                if depth is None:
                    warnings.warn(
                        f"Failed to load depth frame (corrupted?): {frame_path}, "
                        "using fallback"
                    )
                    if frames:
                        # Use previous frame as fallback
                        frames.append(frames[-1].copy())
                    else:
                        # First frame is corrupted - try to find next valid frame
                        fallback_depth = None
                        for j in range(i + 1, seq_len):
                            next_frame_path = osp.join(base_dir, f"{start_frame + j:06d}.png")
                            if osp.exists(next_frame_path):
                                fallback_depth = cv2.imread(next_frame_path, cv2.IMREAD_ANYDEPTH)
                                if fallback_depth is not None:
                                    break
                        if fallback_depth is not None:
                            if self.unit == "m":
                                fallback_depth = fallback_depth.astype(np.float32) / 1000.0
                            else:
                                fallback_depth = fallback_depth.astype(np.float32)
                            frames.append(fallback_depth)
                        else:
                            # Create a zero placeholder as last resort (512x512 depth)
                            frames.append(np.zeros((512, 512), dtype=np.float32))
                    continue
                
                # Convert to float32 and scale to meters
                depth = depth.astype(np.float32)
                # Depth is typically in millimeters, convert to meters
                if self.unit == "m":
                    depth = depth / 1000.0
                    
                frames.append(depth)
            else:
                # Handle missing frames
                warnings.warn(f"Depth frame not found: {frame_path}, using fallback")
                if frames:
                    # Repeat last frame
                    frames.append(frames[-1].copy())
                else:
                    # First frame is missing - try to find next valid frame
                    fallback_depth = None
                    for j in range(i + 1, seq_len):
                        next_frame_path = osp.join(base_dir, f"{start_frame + j:06d}.png")
                        if osp.exists(next_frame_path):
                            fallback_depth = cv2.imread(next_frame_path, cv2.IMREAD_ANYDEPTH)
                            if fallback_depth is not None:
                                break
                    if fallback_depth is not None:
                        if self.unit == "m":
                            fallback_depth = fallback_depth.astype(np.float32) / 1000.0
                        else:
                            fallback_depth = fallback_depth.astype(np.float32)
                        frames.append(fallback_depth)
                    else:
                        # Create a zero placeholder as last resort (512x512 depth)
                        frames.append(np.zeros((512, 512), dtype=np.float32))
        
        return frames

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_info = self.data_list[index].copy()
        
        # If not using all pairs, randomly select camera pair
        if not self.use_all_pairs:
            data_info['rgb_camera'] = random.choice(self.rgb_cameras)
            data_info['depth_camera'] = random.choice(self.depth_cameras)

            if self.colocated:
                # Ensure selected cameras are colocated
                if data_info['rgb_camera'] != data_info['depth_camera']:
                    # If not colocated, set depth camera to rgb camera
                    data_info['depth_camera'] = data_info['rgb_camera']
        
        # Load camera parameters
        cameras = self._load_camera_params(data_info['seq_dir'])
        
        # Load SMPL parameters
        smpl_params = self._load_smpl_params(data_info['seq_dir'])
        
        # Load precomputed 3D keypoints and vertices
        precomputed_data = self._load_keypoints_3d(data_info['seq_dir'])
        
        # Get ground truth SMPL parameters for this sequence
        # Select frame based on causal or non-causal setting
        if self.causal:
            gt_frame_idx = data_info['start_frame'] + self.seq_len - 1
        else:
            middle_offset = self.seq_len // 2
            gt_frame_idx = data_info['start_frame'] + middle_offset
        
        # Ensure frame index is within bounds
        gt_frame_idx = min(gt_frame_idx, smpl_params['global_orient'].shape[0] - 1)
        
        gt_smpl = {
            'global_orient': smpl_params['global_orient'][gt_frame_idx],
            'body_pose': smpl_params['body_pose'][gt_frame_idx],
            'betas': smpl_params['betas'][gt_frame_idx],
            'transl': smpl_params['transl'][gt_frame_idx],
        }
        
        # Convert transl to meters if specified
        if self.unit == "m":
            gt_smpl['transl'] = gt_smpl['transl']  # Already in meters
        
        # Get ground truth 3D keypoints and vertices
        gt_keypoints = None
        gt_vertices = None
        
        if precomputed_data is not None:
            if 'keypoints_3d' in precomputed_data:
                gt_keypoints = precomputed_data['keypoints_3d'][gt_frame_idx]  # (24, 3) - SMPL joints
                # Convert to meters if specified (should already be in meters from SMPL)
                if self.unit == "m":
                    gt_keypoints = gt_keypoints  # Already in meters
            
            if 'vertices' in precomputed_data:
                gt_vertices = precomputed_data['vertices'][gt_frame_idx]  # (6890, 3) - SMPL vertices
                # Convert to meters if specified (should already be in meters from SMPL)
                if self.unit == "m":
                    gt_vertices = gt_vertices  # Already in meters
        
        sample = {
            "gt_smpl": gt_smpl,
            "gt_keypoints": gt_keypoints,  # (24, 3) SMPL joints or None if not precomputed
            "gt_vertices": gt_vertices,  # (6890, 3) SMPL vertices or None if not precomputed
            "sample_id": f"{data_info['seq_name']}_rgb_{data_info['rgb_camera']}_depth_{data_info['depth_camera']}_{data_info['start_frame']}",
            "modalities": list(self.modality_names),
            "anchor_key": self.anchor_key,
        }
        
        # Load RGB frames and camera parameters
        if "rgb" in self.modality_names:
            rgb_frames = self._load_rgb_frames(
                data_info['seq_dir'],
                data_info['rgb_camera'],
                data_info['start_frame'],
                self.seq_len
            )
            sample["input_rgb"] = rgb_frames
            
            # Get RGB camera parameters
            if data_info['rgb_camera'].startswith("kinect"):
                cam_key = f"kinect_color_{data_info['rgb_camera'].split('_')[1]}"
            else:
                cam_key = "iphone"
            
            if cam_key in cameras:
                cam_params = cameras[cam_key]
                K = np.array(cam_params['K'], dtype=np.float32)
                R = np.array(cam_params['R'], dtype=np.float32)
                # HUMMAN camera T values are already in meters
                T = np.array(cam_params['T'], dtype=np.float32).reshape(3, 1)
                
                rgb_camera_dict = {
                    "intrinsic": K,
                    "extrinsic": np.hstack((R, T)).astype(np.float32),
                }
                sample["rgb_camera"] = rgb_camera_dict
        
        # Load depth frames and camera parameters
        if "depth" in self.modality_names:
            depth_frames = self._load_depth_frames(
                data_info['seq_dir'],
                data_info['depth_camera'],
                data_info['start_frame'],
                self.seq_len
            )
            sample["input_depth"] = depth_frames
            
            # Get depth camera parameters
            if data_info['depth_camera'].startswith("kinect"):
                cam_key = f"kinect_depth_{data_info['depth_camera'].split('_')[1]}"
            else:
                cam_key = "iphone"
            
            if cam_key in cameras:
                cam_params = cameras[cam_key]
                K = np.array(cam_params['K'], dtype=np.float32)
                R = np.array(cam_params['R'], dtype=np.float32)
                # HUMMAN camera T values are already in meters
                T = np.array(cam_params['T'], dtype=np.float32).reshape(3, 1)
                
                depth_camera_dict = {
                    "intrinsic": K,
                    "extrinsic": np.hstack((R, T)).astype(np.float32),
                }
                sample["depth_camera"] = depth_camera_dict
        
        # Transform extrinsics to anchor coordinate system
        if "rgb" in self.modality_names and "depth" in self.modality_names:
            rgb_extrinsic = sample["rgb_camera"]["extrinsic"]  # 3x4
            depth_extrinsic = sample["depth_camera"]["extrinsic"]  # 3x4
            
            # Extract R and T from [R|T] format
            rgb_R, rgb_T = rgb_extrinsic[:, :3], rgb_extrinsic[:, 3:]
            depth_R, depth_T = depth_extrinsic[:, :3], depth_extrinsic[:, 3:]
            
            if self.anchor_key == "input_rgb":
                # Set RGB camera to identity (anchor)
                sample["rgb_camera"]["extrinsic"] = np.eye(3, 4, dtype=np.float32)
                
                # Transform depth camera relative to RGB
                R_rel = depth_R @ np.linalg.inv(rgb_R)
                T_rel = depth_T - R_rel @ rgb_T
                sample["depth_camera"]["extrinsic"] = np.hstack([R_rel, T_rel]).astype(np.float32)
                
                # Transform SMPL parameters to RGB camera space
                # The SMPL transl is in world coordinates, need to transform to RGB camera space
                transl_world = gt_smpl['transl'].reshape(3, 1)
                transl_rgb = rgb_R @ transl_world + rgb_T
                sample["gt_smpl"]['transl'] = transl_rgb.flatten()
                global_orient_world = np.asarray(gt_smpl['global_orient'], dtype=np.float32)
                R_smpl = axis_angle_to_matrix_np(global_orient_world)
                R_smpl_rgb = rgb_R @ R_smpl
                sample["gt_smpl"]['global_orient'] = matrix_to_axis_angle_np(R_smpl_rgb)
                
                # Transform gt_keypoints to RGB camera space
                if gt_keypoints is not None:
                    keypoints_world = gt_keypoints.T  # (3, 24)
                    keypoints_rgb = rgb_R @ keypoints_world + rgb_T
                    sample["gt_keypoints"] = keypoints_rgb.T  # (24, 3)
                
                # Transform gt_vertices to RGB camera space
                if gt_vertices is not None:
                    vertices_world = gt_vertices.T  # (3, 6890)
                    vertices_rgb = rgb_R @ vertices_world + rgb_T
                    sample["gt_vertices"] = vertices_rgb.T  # (6890, 3)
                
            elif self.anchor_key == "input_depth":
                # Set depth camera to identity (anchor)
                sample["depth_camera"]["extrinsic"] = np.eye(3, 4, dtype=np.float32)
                
                # Transform RGB camera relative to depth
                R_rel = rgb_R @ np.linalg.inv(depth_R)
                T_rel = rgb_T - R_rel @ depth_T
                sample["rgb_camera"]["extrinsic"] = np.hstack([R_rel, T_rel]).astype(np.float32)
                
                # Transform SMPL parameters to depth camera space
                transl_world = gt_smpl['transl'].reshape(3, 1)
                transl_depth = depth_R @ transl_world + depth_T
                sample["gt_smpl"]['transl'] = transl_depth.flatten()
                global_orient_world = np.asarray(gt_smpl['global_orient'], dtype=np.float32)
                R_smpl = axis_angle_to_matrix_np(global_orient_world)
                R_smpl_depth = depth_R @ R_smpl
                sample["gt_smpl"]['global_orient'] = matrix_to_axis_angle_np(R_smpl_depth)
                
                # Transform gt_keypoints to depth camera space
                if gt_keypoints is not None:
                    keypoints_world = gt_keypoints.T  # (3, 24)
                    keypoints_depth = depth_R @ keypoints_world + depth_T
                    sample["gt_keypoints"] = keypoints_depth.T  # (24, 3)
                
                # Transform gt_vertices to depth camera space
                if gt_vertices is not None:
                    vertices_world = gt_vertices.T  # (3, 6890)
                    vertices_depth = depth_R @ vertices_world + depth_T
                    sample["gt_vertices"] = vertices_depth.T  # (6890, 3)
        
        elif self.anchor_key == "input_rgb" and "rgb" in self.modality_names:
            # Only RGB, set to identity
            sample["rgb_camera"]["extrinsic"] = np.eye(3, 4, dtype=np.float32)
            
            # Transform SMPL to RGB camera space
            rgb_extrinsic = sample["rgb_camera"]["extrinsic"]
            rgb_R, rgb_T = rgb_extrinsic[:, :3], rgb_extrinsic[:, 3:]
            transl_world = gt_smpl['transl'].reshape(3, 1)
            transl_rgb = rgb_R @ transl_world + rgb_T
            sample["gt_smpl"]['transl'] = transl_rgb.flatten()
            global_orient_world = np.asarray(gt_smpl['global_orient'], dtype=np.float32)
            R_smpl = axis_angle_to_matrix_np(global_orient_world)
            R_smpl_rgb = rgb_R @ R_smpl
            sample["gt_smpl"]['global_orient'] = matrix_to_axis_angle_np(R_smpl_rgb)
            
            # Transform gt_keypoints to RGB camera space
            if gt_keypoints is not None:
                keypoints_world = gt_keypoints.T  # (3, 24)
                keypoints_rgb = rgb_R @ keypoints_world + rgb_T
                sample["gt_keypoints"] = keypoints_rgb.T  # (24, 3)
            
            # Transform gt_vertices to RGB camera space
            if gt_vertices is not None:
                vertices_world = gt_vertices.T  # (3, 6890)
                vertices_rgb = rgb_R @ vertices_world + rgb_T
                sample["gt_vertices"] = vertices_rgb.T  # (6890, 3)
            
        elif self.anchor_key == "input_depth" and "depth" in self.modality_names:
            # Only depth, set to identity
            sample["depth_camera"]["extrinsic"] = np.eye(3, 4, dtype=np.float32)
            
            # Transform SMPL to depth camera space
            depth_extrinsic = sample["depth_camera"]["extrinsic"]
            depth_R, depth_T = depth_extrinsic[:, :3], depth_extrinsic[:, 3:]
            transl_world = gt_smpl['transl'].reshape(3, 1)
            transl_depth = depth_R @ transl_world + depth_T
            sample["gt_smpl"]['transl'] = transl_depth.flatten()
            global_orient_world = np.asarray(gt_smpl['global_orient'], dtype=np.float32)
            R_smpl = axis_angle_to_matrix_np(global_orient_world)
            R_smpl_depth = depth_R @ R_smpl
            sample["gt_smpl"]['global_orient'] = matrix_to_axis_angle_np(R_smpl_depth)
            
            # Transform gt_keypoints to depth camera space
            if gt_keypoints is not None:
                keypoints_world = gt_keypoints.T  # (3, 24)
                keypoints_depth = depth_R @ keypoints_world + depth_T
                sample["gt_keypoints"] = keypoints_depth.T  # (24, 3)
            
            # Transform gt_vertices to depth camera space
            if gt_vertices is not None:
                vertices_world = gt_vertices.T  # (3, 6890)
                vertices_depth = depth_R @ vertices_world + depth_T
                sample["gt_vertices"] = vertices_depth.T  # (6890, 3)
        
        # Apply pipeline transforms
        sample = self.pipeline(sample)
        
        return sample


class HummanPreprocessedDataset(BaseDataset):
    def __init__(
        self,
        data_root: str,
        unit: str = "m",
        pipeline: List[dict] = [],
        split: str = "train",
        modality_names: Sequence[str] = ["rgb", "depth"],
        rgb_cameras: Optional[Sequence[str]] = None,
        depth_cameras: Optional[Sequence[str]] = None,
        anchor_key: str = "input_rgb",
        seq_len: int = 5,
        seq_step: int = 1,
        pad_seq: bool = False,
        causal: bool = False,
        use_all_pairs: bool = False,
        random_seed: Optional[int] = None,
        max_samples: Optional[int] = None,
        colocated: bool = False,
    ):
        super().__init__(pipeline=pipeline)
        self.data_root = data_root
        self.split = split
        self.unit = unit
        self.seq_len = seq_len
        self.seq_step = seq_step
        self.causal = causal
        self.pad_seq = pad_seq
        self.modality_names = modality_names
        self.use_all_pairs = use_all_pairs
        self.random_seed = random_seed
        self.max_samples = max_samples
        self.colocated = colocated

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.available_kinect_cameras = [f"kinect_{i:03d}" for i in range(10)]
        self.available_iphone_cameras = ["iphone"]

        if rgb_cameras is None:
            self.rgb_cameras = self.available_kinect_cameras + self.available_iphone_cameras
        else:
            self.rgb_cameras = list(rgb_cameras)

        if depth_cameras is None:
            self.depth_cameras = self.available_kinect_cameras + self.available_iphone_cameras
        else:
            self.depth_cameras = list(depth_cameras)

        valid_modalities = {"rgb", "depth"}
        invalid_modalities = set(modality_names) - valid_modalities
        if invalid_modalities:
            warnings.warn(
                f"Invalid modality names detected: {invalid_modalities}. "
                f"Only 'rgb' and 'depth' are supported for Humman dataset."
            )

        valid_anchor_keys = {f"input_{mod}" for mod in modality_names}
        if anchor_key not in valid_anchor_keys:
            warnings.warn(
                f"Invalid anchor_key: {anchor_key}. "
                f"Must be one of {valid_anchor_keys}. "
                f"Defaulting to 'input_rgb'."
            )
            self.anchor_key = "input_rgb" if "rgb" in modality_names else f"input_{modality_names[0]}"
        else:
            self.anchor_key = anchor_key

        if unit not in {"mm", "m"}:
            warnings.warn(
                f"Invalid unit: {unit}. Defaulting to 'm'."
            )
            self.unit = "m"

        self._seq_re = re.compile(r"(p\d+_a\d+)")
        self._cam_re = re.compile(r"(kinect_\d{3}|iphone)")
        self._frame_re = re.compile(r"(\d+)$")

        self.file_index = self._index_files()
        self.data_list = self._build_dataset()
        if self.max_samples is not None:
            if self.max_samples <= 0:
                self.data_list = []
            else:
                rng = np.random.RandomState(self.random_seed if self.random_seed is not None else 0)
                indices = rng.permutation(len(self.data_list))[:self.max_samples]
                self.data_list = [self.data_list[i] for i in indices]

    def _index_files(self):
        file_index = {m: {} for m in self.modality_names}
        for modality in self.modality_names:
            modality_dir = osp.join(self.data_root, modality)
            if not osp.exists(modality_dir):
                continue
            for fn in os.listdir(modality_dir):
                name, _ = osp.splitext(fn)
                seq_match = self._seq_re.search(name)
                cam_match = self._cam_re.search(name)
                frame_match = self._frame_re.search(name)
                if not (seq_match and cam_match and frame_match):
                    continue
                seq_name = seq_match.group(1)
                camera = cam_match.group(1)
                frame_idx = int(frame_match.group(1))
                file_index[modality].setdefault(seq_name, {}).setdefault(camera, []).append(
                    (frame_idx, osp.join(modality_dir, fn))
                )

        for modality in file_index:
            for seq_name in file_index[modality]:
                for camera in file_index[modality][seq_name]:
                    file_index[modality][seq_name][camera].sort(key=lambda x: x[0])

        return file_index

    def _build_dataset(self):
        data_list = []

        seq_names = set()
        for modality in self.modality_names:
            seq_names.update(self.file_index.get(modality, {}).keys())
        seq_names = sorted(seq_names)

        person_ids = sorted(list(set([s.split("_")[0] for s in seq_names])))
        split_idx = int(len(person_ids) * 0.8)

        if self.split == "train":
            valid_persons = set(person_ids[:split_idx])
        elif self.split == "test":
            valid_persons = set(person_ids[split_idx:])
        elif self.split == "train_mini":
            valid_persons = set(person_ids[:16])
        elif self.split == "test_mini":
            valid_persons = set(person_ids[split_idx:split_idx+4])
        else:
            valid_persons = set(person_ids)

        for seq_name in seq_names:
            person_id = seq_name.split("_")[0]
            if person_id not in valid_persons:
                continue

            rgb_cams = list(self.file_index.get("rgb", {}).get(seq_name, {}).keys())
            depth_cams = list(self.file_index.get("depth", {}).get(seq_name, {}).keys())
            # Filter cameras to only include those requested
            if self.rgb_cameras:
                rgb_cams = [c for c in rgb_cams if c in self.rgb_cameras]
            if self.depth_cameras:
                depth_cams = [c for c in depth_cams if c in self.depth_cameras]

            if "rgb" in self.modality_names and not rgb_cams:
                continue
            if "depth" in self.modality_names and not depth_cams:
                continue

            ref_modality = self.modality_names[0]
            ref_cams = list(self.file_index.get(ref_modality, {}).get(seq_name, {}).keys())
            if not ref_cams:
                continue
            ref_frames = self.file_index[ref_modality][seq_name][ref_cams[0]]
            num_frames = len(ref_frames)
            if num_frames < self.seq_len:
                continue

            for start_idx in range(0, num_frames - self.seq_len + 1, self.seq_step):
                if self.use_all_pairs:
                    for rgb_cam in rgb_cams:
                        for depth_cam in depth_cams:
                            data_list.append({
                                "seq_name": seq_name,
                                "person_id": person_id,
                                "start_frame": start_idx,
                                "num_frames": num_frames,
                                "rgb_camera": rgb_cam,
                                "depth_camera": depth_cam,
                                "rgb_cameras": rgb_cams,
                                "depth_cameras": depth_cams,
                            })
                else:
                    data_list.append({
                        "seq_name": seq_name,
                        "person_id": person_id,
                        "start_frame": start_idx,
                        "num_frames": num_frames,
                        "rgb_camera": None,
                        "depth_camera": None,
                        "rgb_cameras": rgb_cams,
                        "depth_cameras": depth_cams,
                    })

        return data_list

    def _load_camera_params(self, seq_name):
        camera_file = osp.join(self.data_root, "cameras", f"{seq_name}_cameras.json")
        with open(camera_file, "r") as f:
            cameras = json.load(f)
        return cameras

    def _load_smpl_params(self, seq_name):
        smpl_file = osp.join(self.data_root, "smpl", f"{seq_name}_smpl_params.npz")
        smpl_data = np.load(smpl_file)
        return {
            "global_orient": smpl_data["global_orient"],
            "body_pose": smpl_data["body_pose"],
            "betas": smpl_data["betas"],
            "transl": smpl_data["transl"],
        }

    def _load_keypoints_3d(self, seq_name):
        keypoints_file = osp.join(self.data_root, "skl", f"{seq_name}_keypoints_3d.npz")
        if osp.exists(keypoints_file):
            keypoints_data = np.load(keypoints_file)
            return keypoints_data["keypoints_3d"]
        warnings.warn(
            f"keypoints_3d.npz not found for {seq_name}. "
            f"Run tools/generate_humman_keypoints.py to precompute keypoints."
        )
        return None

    def _load_rgb_frames(self, seq_name, camera_name, start_frame):
        frame_list = self.file_index["rgb"][seq_name][camera_name]
        frames = []
        for i in range(self.seq_len):
            idx = start_frame + i
            idx = min(idx, len(frame_list) - 1)
            frame_path = frame_list[idx][1]
            frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            if frame is None:
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((512, 512, 3), dtype=np.uint8))
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        return frames

    def _load_depth_frames(self, seq_name, camera_name, start_frame):
        frame_list = self.file_index["depth"][seq_name][camera_name]
        frames = []
        for i in range(self.seq_len):
            idx = start_frame + i
            idx = min(idx, len(frame_list) - 1)
            frame_path = frame_list[idx][1]
            depth = cv2.imread(frame_path, cv2.IMREAD_ANYDEPTH)
            if depth is None:
                depth = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((512, 512), dtype=np.float32))
                continue
            depth = depth.astype(np.float32)
            if self.unit == "m":
                depth = depth / 1000.0
            frames.append(depth)
        return frames

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_info = self.data_list[index].copy()

        if not self.use_all_pairs:
            if data_info["rgb_camera"] is None and "rgb" in self.modality_names:
                data_info["rgb_camera"] = random.choice(data_info["rgb_cameras"])
            if data_info["depth_camera"] is None and "depth" in self.modality_names:
                data_info["depth_camera"] = random.choice(data_info["depth_cameras"])

            if self.colocated and "rgb" in self.modality_names and "depth" in self.modality_names:
                if data_info["rgb_camera"] != data_info["depth_camera"]:
                    data_info["depth_camera"] = data_info["rgb_camera"]       

        cameras = self._load_camera_params(data_info["seq_name"])
        smpl_params = self._load_smpl_params(data_info["seq_name"])
        keypoints_3d = self._load_keypoints_3d(data_info["seq_name"])

        if self.causal:
            gt_frame_idx = data_info["start_frame"] + self.seq_len - 1
        else:
            middle_offset = self.seq_len // 2
            gt_frame_idx = data_info["start_frame"] + middle_offset

        gt_frame_idx = min(gt_frame_idx, smpl_params["global_orient"].shape[0] - 1)

        gt_smpl = {
            "global_orient": smpl_params["global_orient"][gt_frame_idx],
            "body_pose": smpl_params["body_pose"][gt_frame_idx],
            "betas": smpl_params["betas"][gt_frame_idx],
            "transl": smpl_params["transl"][gt_frame_idx],
        }

        gt_keypoints = keypoints_3d[gt_frame_idx] if keypoints_3d is not None else None

        sample = {
            "gt_smpl": gt_smpl,
            "gt_keypoints": gt_keypoints,
            "sample_id": (
                f"{data_info['seq_name']}_rgb_{data_info['rgb_camera']}_"
                f"depth_{data_info['depth_camera']}_{data_info['start_frame']}"
            ),
            "modalities": list(self.modality_names),
            "anchor_key": self.anchor_key,
        }

        if "rgb" in self.modality_names:
            rgb_frames = self._load_rgb_frames(
                data_info["seq_name"],
                data_info["rgb_camera"],
                data_info["start_frame"],
            )
            sample["input_rgb"] = rgb_frames
            if data_info["rgb_camera"].startswith("kinect"):
                cam_key = f"kinect_color_{data_info['rgb_camera'].split('_')[1]}"
            else:
                cam_key = "iphone"
            if cam_key in cameras:
                cam_params = cameras[cam_key]
                K = np.array(cam_params["K"], dtype=np.float32)
                R = np.array(cam_params["R"], dtype=np.float32)
                T = np.array(cam_params["T"], dtype=np.float32).reshape(3, 1)
                sample["rgb_camera"] = {
                    "intrinsic": K,
                    "extrinsic": np.hstack((R, T)).astype(np.float32),
                }

        if "depth" in self.modality_names:
            depth_frames = self._load_depth_frames(
                data_info["seq_name"],
                data_info["depth_camera"],
                data_info["start_frame"],
            )
            sample["input_depth"] = depth_frames
            if data_info["depth_camera"].startswith("kinect"):
                cam_key = f"kinect_depth_{data_info['depth_camera'].split('_')[1]}"
            else:
                cam_key = "iphone"
            if cam_key in cameras:
                cam_params = cameras[cam_key]
                K = np.array(cam_params["K"], dtype=np.float32)
                R = np.array(cam_params["R"], dtype=np.float32)
                T = np.array(cam_params["T"], dtype=np.float32).reshape(3, 1)
                sample["depth_camera"] = {
                    "intrinsic": K,
                    "extrinsic": np.hstack((R, T)).astype(np.float32),
                }

        if "rgb" in self.modality_names and "depth" in self.modality_names:
            rgb_extrinsic = sample["rgb_camera"]["extrinsic"]
            depth_extrinsic = sample["depth_camera"]["extrinsic"]
            rgb_R, rgb_T = rgb_extrinsic[:, :3], rgb_extrinsic[:, 3:]
            depth_R, depth_T = depth_extrinsic[:, :3], depth_extrinsic[:, 3:]

            if self.anchor_key == "input_rgb":
                sample["rgb_camera"]["extrinsic"] = np.eye(3, 4, dtype=np.float32)
                R_rel = depth_R @ np.linalg.inv(rgb_R)
                T_rel = depth_T - R_rel @ rgb_T
                sample["depth_camera"]["extrinsic"] = np.hstack([R_rel, T_rel]).astype(np.float32)

                transl_world = gt_smpl["transl"].reshape(3, 1)
                transl_rgb = rgb_R @ transl_world + rgb_T
                sample["gt_smpl"]["transl"] = transl_rgb.flatten()
                global_orient_world = np.asarray(gt_smpl["global_orient"], dtype=np.float32)
                R_smpl = axis_angle_to_matrix_np(global_orient_world)
                R_smpl_rgb = rgb_R @ R_smpl
                sample["gt_smpl"]["global_orient"] = matrix_to_axis_angle_np(R_smpl_rgb)

                if gt_keypoints is not None:
                    keypoints_world = gt_keypoints.T
                    keypoints_rgb = rgb_R @ keypoints_world + rgb_T
                    sample["gt_keypoints"] = keypoints_rgb.T

            elif self.anchor_key == "input_depth":
                sample["depth_camera"]["extrinsic"] = np.eye(3, 4, dtype=np.float32)
                R_rel = rgb_R @ np.linalg.inv(depth_R)
                T_rel = rgb_T - R_rel @ depth_T
                sample["rgb_camera"]["extrinsic"] = np.hstack([R_rel, T_rel]).astype(np.float32)

                transl_world = gt_smpl["transl"].reshape(3, 1)
                transl_depth = depth_R @ transl_world + depth_T
                sample["gt_smpl"]["transl"] = transl_depth.flatten()
                global_orient_world = np.asarray(gt_smpl["global_orient"], dtype=np.float32)
                R_smpl = axis_angle_to_matrix_np(global_orient_world)
                R_smpl_depth = depth_R @ R_smpl
                sample["gt_smpl"]["global_orient"] = matrix_to_axis_angle_np(R_smpl_depth)

                if gt_keypoints is not None:
                    keypoints_world = gt_keypoints.T
                    keypoints_depth = depth_R @ keypoints_world + depth_T
                    sample["gt_keypoints"] = keypoints_depth.T

        elif self.anchor_key == "input_rgb" and "rgb" in self.modality_names:
            sample["rgb_camera"]["extrinsic"] = np.eye(3, 4, dtype=np.float32)
            rgb_extrinsic = sample["rgb_camera"]["extrinsic"]
            rgb_R, rgb_T = rgb_extrinsic[:, :3], rgb_extrinsic[:, 3:]
            transl_world = gt_smpl["transl"].reshape(3, 1)
            transl_rgb = rgb_R @ transl_world + rgb_T
            sample["gt_smpl"]["transl"] = transl_rgb.flatten()
            global_orient_world = np.asarray(gt_smpl["global_orient"], dtype=np.float32)
            R_smpl = axis_angle_to_matrix_np(global_orient_world)
            R_smpl_rgb = rgb_R @ R_smpl
            sample["gt_smpl"]["global_orient"] = matrix_to_axis_angle_np(R_smpl_rgb)
            if gt_keypoints is not None:
                keypoints_world = gt_keypoints.T
                keypoints_rgb = rgb_R @ keypoints_world + rgb_T
                sample["gt_keypoints"] = keypoints_rgb.T

        elif self.anchor_key == "input_depth" and "depth" in self.modality_names:
            sample["depth_camera"]["extrinsic"] = np.eye(3, 4, dtype=np.float32)
            depth_extrinsic = sample["depth_camera"]["extrinsic"]
            depth_R, depth_T = depth_extrinsic[:, :3], depth_extrinsic[:, 3:]
            transl_world = gt_smpl["transl"].reshape(3, 1)
            transl_depth = depth_R @ transl_world + depth_T
            sample["gt_smpl"]["transl"] = transl_depth.flatten()
            global_orient_world = np.asarray(gt_smpl["global_orient"], dtype=np.float32)
            R_smpl = axis_angle_to_matrix_np(global_orient_world)
            R_smpl_depth = depth_R @ R_smpl
            sample["gt_smpl"]["global_orient"] = matrix_to_axis_angle_np(R_smpl_depth)
            if gt_keypoints is not None:
                keypoints_world = gt_keypoints.T
                keypoints_depth = depth_R @ keypoints_world + depth_T
                sample["gt_keypoints"] = keypoints_depth.T

        sample = self.pipeline(sample)
        return sample
