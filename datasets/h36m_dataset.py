import os
import os.path as osp
import cv2
import numpy as np
import yaml
import pickle
import glob
import cdflib
from typing import Callable, List, Optional, Sequence, Tuple, Union

from datasets.base_dataset import BaseDataset

class H36MDataset(BaseDataset):
    def __init__(self, pipeline: List[dict] = [], h36m_root: str = '', split: str = 'train', 
                 sequence_length: int = 16, frame_step: int = 1):
        super().__init__(pipeline=pipeline)
        self.h36m_root = h36m_root
        self.split = split
        self.sequence_length = sequence_length
        self.frame_step = frame_step
        
        # Dataset paths
        self.images_root = '/data/yzhanghe/H36M-Toolbox/images'
        self.extracted_root = '/data/yzhanghe/H36M-Toolbox/extracted'
        
        # H3.6M protocol splits
        if split == 'train':
            self.subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
        elif split == 'test':
            self.subjects = ['S9', 'S11']
        else:
            self.subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
            
        # Action mapping - H3.6M actions
        self.action_names = {
            2: 'Directions', 3: 'Discussion', 4: 'Eating', 5: 'Greeting',
            6: 'Phoning', 7: 'Photo', 8: 'Posing', 9: 'Purchases',
            10: 'Sitting', 11: 'SittingDown', 12: 'Smoking', 13: 'Waiting',
            14: 'WalkDog', 15: 'Walking', 16: 'WalkTogether'
        }
        
        # Camera IDs mapping (1-indexed to names)
        self.camera_ids = ['54138969', '55011271', '58860488', '60457274']
        
        # Build dataset index
        self.data_list = self._build_dataset()
        
        # Load camera parameters
        self.camera_params = self._load_camera_params()

    def _build_dataset(self):
        """Build dataset index with available sequences."""
        data_list = []
        
        for subject in self.subjects:
            # Get available image sequences for this subject
            subject_dirs = glob.glob(osp.join(self.images_root, f's_{subject[1:].zfill(2)}_act_*'))
            
            for seq_dir in subject_dirs:
                seq_name = osp.basename(seq_dir)
                parts = seq_name.split('_')
                if len(parts) >= 6:
                    subject_id = parts[1]
                    action_id = parts[3]
                    subaction_id = parts[5]
                    camera_id = parts[7] if len(parts) > 7 else parts[6]
                    
                    # Count available frames
                    frame_files = glob.glob(osp.join(seq_dir, '*.jpg'))
                    num_frames = len(frame_files)
                    
                    if num_frames >= self.sequence_length:
                        # Create sequences with overlap
                        for start_idx in range(0, num_frames - self.sequence_length + 1, self.frame_step):
                            data_info = {
                                'subject': f'S{int(subject_id)}',  # Convert to int to remove zero padding
                                'action': action_id,
                                'subaction': subaction_id,
                                'camera': camera_id,
                                'seq_dir': seq_dir,
                                'start_frame': start_idx,
                                'num_frames': num_frames
                            }
                            data_list.append(data_info)
        
        return data_list
    
    def _load_camera_params(self):
        """Load camera parameters for all subjects."""
        camera_params = {}
        
        for subject in self.subjects:
            calib_file = osp.join(self.extracted_root, subject, 'calibration.toml')
            if osp.exists(calib_file):
                # Parse TOML-like format manually
                cameras = {}
                with open(calib_file, 'r') as f:
                    content = f.read()
                    
                # Extract camera data (simple parsing)
                import re
                cam_blocks = re.findall(r'\[cam_(\d+)\]([^\[]*)', content)
                for cam_idx, cam_data in cam_blocks:
                    # Extract camera name
                    name_match = re.search(r'name = "(\d+)"', cam_data)
                    if name_match:
                        cam_name = name_match.group(1)
                        
                        # Extract matrix (intrinsics)
                        matrix_match = re.search(r'matrix = \[\s*\[([^\]]+)\]', cam_data)
                        if matrix_match:
                            matrix_str = matrix_match.group(1)
                            fx, _, cx = [float(x.strip(' ,')) for x in matrix_str.split(',')[:3]]
                            
                            # Get fy and cy from second row
                            fy_match = re.search(r'\[\s*[^,]+,\s*([^,]+),\s*([^,]+),\s*\]', cam_data)
                            if fy_match:
                                fy = float(fy_match.group(1))
                                cy = float(fy_match.group(2))
                                
                                cameras[cam_name] = {
                                    'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy
                                }
                
                camera_params[subject] = cameras
        
        return camera_params
    
    def _load_pose_data(self, subject, action_name):
        """Load 3D pose data for a sequence."""
        pose_file = osp.join(self.extracted_root, subject, 'MyPoseFeatures', 'D3_Positions', f'{action_name}.cdf')
        if osp.exists(pose_file):
            cdf = cdflib.CDF(pose_file)
            pose_data = cdf.varget('Pose')  # Shape: (32, num_frames, 96)
            # Reshape to (num_frames, 32, 3)
            pose_data = pose_data.transpose(1, 0, 2).reshape(pose_data.shape[1], 32, 3)
            return pose_data
        return None
    
    def _load_depth_data(self, subject, action_name):
        """Load depth data for a sequence."""
        depth_file = osp.join(self.extracted_root, subject, 'TOF', f'{action_name}.cdf')
        if osp.exists(depth_file):
            cdf = cdflib.CDF(depth_file)
            range_frames = cdf.varget('RangeFrames')  # Shape: (1, H, W, num_frames)
            # Transpose to (num_frames, H, W)
            depth_data = range_frames[0].transpose(2, 0, 1)
            return depth_data
        return None
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_info = self.data_list[index]
        
        # Load RGB sequence
        rgb_frames = []
        for i in range(self.sequence_length):
            frame_idx = data_info['start_frame'] + i + 1  # 1-indexed
            frame_name = f"s_{data_info['subject'][1:].zfill(2)}_act_{data_info['action'].zfill(2)}_subact_{data_info['subaction'].zfill(2)}_ca_{data_info['camera'].zfill(2)}_{frame_idx:06d}.jpg"
            frame_path = osp.join(data_info['seq_dir'], frame_name)
            
            if osp.exists(frame_path):
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frames.append(frame)
            else:
                # Handle missing frames by repeating last frame
                if rgb_frames:
                    rgb_frames.append(rgb_frames[-1].copy())
                else:
                    # Create dummy frame if first frame is missing
                    rgb_frames.append(np.zeros((1000, 1000, 3), dtype=np.uint8))
        
        # Load action name for pose/depth data
        action_id = int(data_info['action'])
        action_name = self.action_names.get(action_id, f'Action_{action_id}')
        
        # Load 3D pose data
        pose_data = self._load_pose_data(data_info['subject'], action_name)
        if pose_data is not None:
            # Get poses corresponding to RGB frames
            pose_start = min(data_info['start_frame'], pose_data.shape[0] - self.sequence_length)
            pose_sequence = pose_data[pose_start:pose_start + self.sequence_length]
        else:
            # Raise error if pose data is missing
            raise RuntimeError(f"3D pose data not found for {data_info['subject']} action {action_id} subaction {subaction_id} camera {camera_id}")
        
        # Load depth data  
        depth_data = self._load_depth_data(data_info['subject'], action_name)
        depth_frames = []
        if depth_data is not None:
            # Handle depth frame rate difference (typically half of RGB)
            depth_ratio = depth_data.shape[0] / data_info['num_frames'] if data_info['num_frames'] > 0 else 0.5
            for i in range(self.sequence_length):
                depth_idx = int((data_info['start_frame'] + i) * depth_ratio)
                depth_idx = min(depth_idx, depth_data.shape[0] - 1)
                depth_frames.append(depth_data[depth_idx])
        else:
            # Raise error if depth data is missing
            raise RuntimeError(f"Depth data not found for {data_info['subject']} action {action_id} subaction {subaction_id} camera {camera_id}")
        
        # Get camera parameters
        camera_param = self.camera_params.get(data_info['subject'], {}).get(data_info['camera'], {
            'fx': 1000.0, 'fy': 1000.0, 'cx': 500.0, 'cy': 500.0
        })
        
        sample = {
            'input_rgb': rgb_frames,
            'input_depth': depth_frames,
            'input_rgb_camera': camera_param,
            'input_depth_camera': camera_param,  # Assuming same camera for RGB and depth
            'gt_keypoints': pose_sequence[self.sequence_length // 2],  # Use middle frame pose
            'sample_id': f"{data_info['subject']}_{action_name}_{data_info['subaction']}_{data_info['camera']}_{data_info['start_frame']}",
            'modalities': ['rgb', 'depth'],
            'anchor_key': 'input_rgb'  # Coordinates in RGB camera space
        }
        
        sample = self.pipeline(sample)
        
        return sample


        # TODO: Check coorespondence between rgb, depth, camera params, and 3D pose data
        # TODO: Check depth camera pose.