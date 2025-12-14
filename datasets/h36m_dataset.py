import os
import os.path as osp
import cv2
import numpy as np
import yaml
import pickle
import json
import glob
import cdflib
from typing import Callable, List, Optional, Sequence, Tuple, Union

from datasets.base_dataset import BaseDataset

from .h36m_metadata import load_h36m_metadata
import warnings


class H36MDataset(BaseDataset):
    def __init__(
        self,
        data_root: str = "/data/shared/H36M-Toolbox",
        unit: str = "m",
        pipeline: List[dict] = [],
        split: str = "train",
        modality_names: Sequence[str] = ["rgb", "depth"],
        cameras: Sequence[str] = ['01', '02', '03', '04'], # use all cameras by default
        anchor_key: str = "input_rgb",
        seq_len: int = 5,
        seq_step: int = 1,
        pad_seq: bool = False,
        causal: bool = False,
        remove_static_joints=True
    ):
        super().__init__(pipeline=pipeline)
        self.h36m_root = data_root
        self.split = split
        self.unit = unit
        self.seq_len = seq_len
        self.seq_step = seq_step
        self.causal = causal
        self.remove_static_joints = remove_static_joints
        self.pad_seq = pad_seq
        self.modality_names = modality_names
        self.cameras = cameras
        # Validate modality names
        valid_modalities = {"rgb", "depth"}
        invalid_modalities = set(modality_names) - valid_modalities
        if invalid_modalities:
            warnings.warn(
                f"Invalid modality names detected: {invalid_modalities}. "
                f"Only 'rgb' and 'depth' are supported for H36M dataset."
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

        # Dataset paths
        self.images_root = f"{data_root}/images"
        self.extracted_root = f"{data_root}/extracted"

        # H3.6M protocol splits
        if split == "train":
            self.subjects = ["S1", "S5", "S6", "S7", "S8"]
        elif split == "test":
            self.subjects = ["S9", "S11"]
        elif split == "train_depth":
            self.subjects = ["S1", "S6", "S8", "S9"]
        elif split == "test_depth":
            self.subjects = ["S11"]
        elif split == "train_mini":
            self.subjects = ["S1"]
        elif split == "test_mini":
            self.subjects = ["S11"]
        else:
            self.subjects = ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]

        # Load metadata
        self.metadata = load_h36m_metadata(f"{data_root}/metadata.xml")

        # Build dataset index
        self.data_list = self._build_dataset()

        # Load camera parameters
        self.camera_params = self._load_camera_params()

    def _build_dataset(self):
        """Build dataset index with available sequences."""
        data_list = []

        for subject in self.subjects:
            # Get available image sequences for this subject
            subject_dirs = glob.glob(
                osp.join(self.images_root, f"s_{subject[1:].zfill(2)}_act_*")
            )

            for seq_dir in subject_dirs:
                seq_name = osp.basename(seq_dir)
                parts = seq_name.split("_")
                if len(parts) >= 6:
                    subject_id = parts[1]
                    action_id = parts[3]
                    subaction_id = parts[5]
                    camera_id = parts[7] if len(parts) > 7 else parts[6]

                    # Skip if camera not in selected cameras
                    if camera_id not in self.cameras:
                        continue

                    # Skip S5 SA4 and S7 SA15 (missing depth data)
                    if (subject == "S5" and action_id == "04") or (subject == "S7" and action_id == "15"):
                        continue

                    # Count available frames
                    frame_files = glob.glob(osp.join(seq_dir, "*.jpg"))
                    num_frames = len(frame_files)

                    if num_frames >= self.seq_len:
                        # Create sequences with overlap
                        for start_idx in range(
                            0, num_frames - self.seq_len + 1, self.seq_step
                        ):
                            data_info = {
                                "subject_id": int(subject_id),    
                                "subject": f"S{int(subject_id)}",  # Convert to int to remove zero padding
                                "action": action_id,
                                "subaction": subaction_id,
                                "camera": camera_id,
                                "seq_dir": seq_dir,
                                "start_frame": start_idx,
                                "num_frames": num_frames,
                            }
                            data_list.append(data_info)

        return data_list

    def _load_camera_params(self):
        """Load camera parameters for all subjects from JSON."""
        with open(f"{self.h36m_root}/camera-parameters.json", "r") as f:
            json_data = json.load(f)
        
        # Convert JSON format to dictionary keyed by (subject_id, camera_id)
        # JSON uses standard world-to-camera convention: [R|t] where camera_center = -R.T @ t
        camera_params = {}
        camera_id_map = {
            '54138969': 1,
            '55011271': 2,
            '58860488': 3,
            '60457274': 4
        }
        
        for subject in ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']:
            subject_id = int(subject[1:])
            if subject not in json_data['extrinsics']:
                continue
                
            for cam_name, cam_idx in camera_id_map.items():
                if cam_name not in json_data['extrinsics'][subject]:
                    continue
                    
                # Get extrinsics
                extrinsic_data = json_data['extrinsics'][subject][cam_name]
                R = np.array(extrinsic_data['R'], dtype=np.float32)
                t = np.array(extrinsic_data['t'], dtype=np.float32).flatten()
                
                # Get intrinsics
                intrinsic_data = json_data['intrinsics'][cam_name]
                calib_matrix = np.array(intrinsic_data['calibration_matrix'], dtype=np.float32)
                fx, fy = calib_matrix[0, 0], calib_matrix[1, 1]
                cx, cy = calib_matrix[0, 2], calib_matrix[1, 2]
                distortion = np.array(intrinsic_data['distortion'], dtype=np.float32)
                
                # Store in format: [R, t, f, c, k, p, name]
                # Note: JSON uses standard convention where t is translation, not camera center
                camera_params[(subject_id, cam_idx)] = [
                    R,
                    t.reshape(3, 1),
                    np.array([fx, fy], dtype=np.float32),
                    np.array([cx, cy], dtype=np.float32),
                    distortion[:3],  # radial distortion k1, k2, k3
                    distortion[3:5],  # tangential distortion p1, p2
                    cam_name
                ]
        
        return camera_params

    def _load_pose_data(self, subject, action, subaction, camera):
        """Load 3D pose data for a sequence using the correct approach."""
        # Get the base filename like in the reference script
        basename = self.metadata.get_base_filename(
            subject,
            "{:d}".format(action),
            "{:d}".format(subaction),
            self.metadata.camera_ids[camera - 1],
        )
        annotname = basename + ".cdf"

        # Use D3_Positions_mono_universal (camera coordinates)
        pose_file = osp.join(
            self.extracted_root,
            subject,
            "MyPoseFeatures",
            "D3_Positions_mono_universal",
            annotname,
        )
        if osp.exists(pose_file):
            cdf = cdflib.CDF(pose_file)
            pose_data = cdf.varget(
                "Pose"
            )  # Shape: (32, num_frames, 96) or (num_frames, 32, 3)
            pose_data = np.array(pose_data)

            # Reshape to (num_frames, 32, 3)
            if len(pose_data.shape) == 3 and pose_data.shape[2] == 96:
                # Shape is (32, num_frames, 96)
                pose_data = pose_data.transpose(1, 0, 2).reshape(
                    pose_data.shape[1], 32, 3
                )
            elif len(pose_data.shape) == 3:
                # Already in correct shape (num_frames, 32, 3)
                pass
            else:
                # Fallback reshape
                pose_data = pose_data.reshape(-1, 32, 3)

            return pose_data

        # If not found, throw error
        raise FileNotFoundError(
            f"Pose file not found for subject {subject}, action {action}, subaction {subaction}, camera {camera}"
        )

    def _load_depth_data(self, subject, action, subaction):
        """Load depth data for a sequence."""
        basename = self.metadata.get_base_filename(
            subject,
            "{:d}".format(action),
            "{:d}".format(subaction),
            None
        )
        annotname = basename + ".cdf"
        depth_file = osp.join(self.extracted_root, subject, "TOF", annotname)
        if osp.exists(depth_file):
            cdf = cdflib.CDF(depth_file)
            range_frames = cdf.varget("RangeFrames")  # Shape: (1, H, W, num_frames)
            # Transpose to (num_frames, H, W)
            depth_data = range_frames[0].transpose(2, 0, 1)
            return depth_data
        else:
            # Throw error if depth file not found
            # raise FileNotFoundError(
            #     f"Depth file not found for subject {subject}, action {action}, subaction {subaction}"
            # )
            # TODO: Check depth data missing at S7 action 15 subaction 2
            return None

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_info = self.data_list[index]


        # Load RGB sequence
        if "rgb" in self.modality_names:
            rgb_frames = []
            for i in range(self.seq_len):
                frame_idx = data_info["start_frame"] + i + 1  # 1-indexed
                frame_name = f"s_{data_info['subject'][1:].zfill(2)}_act_{data_info['action'].zfill(2)}_subact_{data_info['subaction'].zfill(2)}_ca_{data_info['camera'].zfill(2)}_{frame_idx:06d}.jpg"
                frame_path = osp.join(data_info["seq_dir"], frame_name)

                if osp.exists(frame_path):
                    frame = cv2.imread(frame_path)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frames.append(frame)
                else:
                    # Handle missing frames by repeating last frame
                    if rgb_frames:
                        rgb_frames.append(rgb_frames[-1].copy())
                    else:
                        # raise error if first frame is missing
                        raise RuntimeError(f"RGB frame not found: {frame_path}")

        # Load action name for pose/depth data
        action_id = int(data_info["action"])
        subaction_id = int(data_info["subaction"])
        camera_id = int(data_info["camera"])

        # Load 3D pose data
        pose_data = self._load_pose_data(
            data_info["subject"], action_id, subaction_id, camera_id
        )
        if pose_data is not None:
            # Get poses corresponding to RGB frames
            pose_start = min(
                data_info["start_frame"], pose_data.shape[0] - self.seq_len
            )
            pose_start = max(0, pose_start)  # Ensure non-negative
            pose_sequence = pose_data[pose_start : pose_start + self.seq_len]

            # If we don't have enough frames, pad with the last frame
            if pose_sequence.shape[0] < self.seq_len:
                last_pose = (
                    pose_sequence[-1:]
                    if pose_sequence.shape[0] > 0
                    else np.zeros((1, 32, 3))
                )
                padding_needed = self.seq_len - pose_sequence.shape[0]
                padding = np.repeat(last_pose, padding_needed, axis=0)
                pose_sequence = np.concatenate([pose_sequence, padding], axis=0)
            
            # Select ground truth pose based on causal or non-causal setting
            if self.causal:
                gt_keypoints = pose_sequence[-1]
            else:
                middle_idx = self.seq_len // 2
                gt_keypoints = pose_sequence[middle_idx]

            # Remove static joints 
            if self.remove_static_joints:
                # Bring the skeleton to 17 joints instead of the original 32
                joints_to_remove = [4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31]
                gt_keypoints = np.delete(gt_keypoints, joints_to_remove, axis=0)
            
            # Convert to meters if specified
            if self.unit == "m":
                gt_keypoints = gt_keypoints / 1000.0

        else:
            # Raise error if pose data is missing
            raise RuntimeError(
                f"3D pose data not found for {data_info['subject']} action {action_id} subaction {subaction_id} camera {camera_id}"  
            )

        # Load depth data
        if "depth" in self.modality_names:    
            depth_data = self._load_depth_data(data_info['subject'], action_id, subaction_id)
            depth_frames = []
            if depth_data is not None:
                # Handle depth frame rate difference (typically half of RGB)
                depth_ratio = depth_data.shape[0] / data_info['num_frames'] if data_info['num_frames'] > 0 else 0.5
                for i in range(self.seq_len):
                    depth_idx = int((data_info['start_frame'] + i) * depth_ratio)
                    depth_idx = min(depth_idx, depth_data.shape[0] - 1)
                    depth_frames.append(depth_data[depth_idx])
            else:
                # Raise error if depth data is missing
                # raise RuntimeError(f"Depth data not found for {data_info['subject']} action {action_id} subaction {subaction_id} camera {camera_id}")
                # If depth data is missing, fill with zeros (H36M depth is 144x176)
                # TODO: Check depth data missing at S7 action 15 subaction 2
                print(f"Warning: Depth data not found for {data_info['subject']} action {action_id} subaction {subaction_id} camera {camera_id}. Filling with zeros.")
                depth_frames = [np.zeros((144, 176), dtype=np.float32) for _ in range(self.seq_len)]

        # Get camera parameters
        if "rgb" in self.modality_names:
            rgb_camera_param = self.camera_params[(data_info['subject_id'], camera_id)]
            # Only expose intrinsic and extrinsic matrices
            R = rgb_camera_param[0]
            T = rgb_camera_param[1]
            fx = float(rgb_camera_param[2][0])
            fy = float(rgb_camera_param[2][1])
            cx = float(rgb_camera_param[3][0])
            cy = float(rgb_camera_param[3][1])

            # Convert translation to meters if specified
            T_scaled = T / 1000.0 if self.unit == "m" else T
            
            rgb_camera_dict = {
                "intrinsic": np.array(
                    [
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1],
                    ],
                    dtype=np.float32,
                ),
                "extrinsic": np.hstack((R, T_scaled.reshape(3, 1))).astype(np.float32),
            }

        # Get depth camera parameters (assume same extrinsics as camera 02 for now, intrinsics calculated manually)
        # TODO: Find correct depth camera parameters if available
        if "depth" in self.modality_names:
            depth_camera_param = self.camera_params[(data_info['subject_id'], 2)]
            R = depth_camera_param[0]
            T = depth_camera_param[1]
            # Depth sensor has fixed intrinsic parameters (different from RGB)
            fx = 220.0
            fy = 231.2
            cx = 88.0
            cy = 72.0
            
            # Convert translation to meters if specified
            T_scaled = T / 1000.0 if self.unit == "m" else T
            
            depth_camera_dict = {
                "intrinsic": np.array(
                    [
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1],
                    ],
                    dtype=np.float32,
                ),
                "extrinsic": np.hstack((R, T_scaled.reshape(3, 1))).astype(np.float32),
            }

        # Transform extrinsics and keypoints to anchor_key coordinate system
        if "depth" in self.modality_names:
            depth_extrinsic = depth_camera_dict["extrinsic"]  # 3x4
            rgb_extrinsic = rgb_camera_dict["extrinsic"]  # 3x4
            
            # Extract R and T from [R|T] format
            depth_R, depth_T = depth_extrinsic[:, :3], depth_extrinsic[:, 3:]
            rgb_R, rgb_T = rgb_extrinsic[:, :3], rgb_extrinsic[:, 3:]
            
            if self.anchor_key == "input_rgb":
                # Keypoints are already in RGB camera space (from D3_Positions_mono_universal)
                # Set RGB camera extrinsic to identity since it's the anchor
                rgb_camera_dict["extrinsic"] = np.eye(3, 4, dtype=np.float32)
                
                # Transform depth camera extrinsic to be relative to RGB camera
                # For p_cam = R @ p_world + T format:
                # p_depth = R_depth @ p_world + T_depth
                # p_rgb = R_rgb @ p_world + T_rgb
                # Therefore: p_depth = R_depth @ inv(R_rgb) @ (p_rgb - T_rgb) + T_depth
                #          = R_depth @ inv(R_rgb) @ p_rgb + (T_depth - R_depth @ inv(R_rgb) @ T_rgb)
                R_rel = depth_R @ np.linalg.inv(rgb_R)
                T_rel = depth_T - R_rel @ rgb_T
                depth_camera_dict["extrinsic"] = np.hstack([R_rel, T_rel]).astype(np.float32)
                
            elif self.anchor_key == "input_depth":
                # Transform keypoints from RGB camera space to depth camera space
                R_rel = depth_R @ np.linalg.inv(rgb_R)
                T_rel = depth_T - R_rel @ rgb_T
                gt_keypoints_transformed = (R_rel @ gt_keypoints.T).T + T_rel.T
                gt_keypoints = gt_keypoints_transformed
                
                # Set depth camera extrinsic to identity since it's the anchor
                depth_camera_dict["extrinsic"] = np.eye(3, 4, dtype=np.float32)
                
                # Transform RGB camera extrinsic to be relative to depth camera
                R_rel_inv = rgb_R @ np.linalg.inv(depth_R)
                T_rel_inv = rgb_T - R_rel_inv @ depth_T
                rgb_camera_dict["extrinsic"] = np.hstack([R_rel_inv, T_rel_inv]).astype(np.float32)
                
        elif self.anchor_key == "input_depth":
            raise RuntimeError("Anchor key set to input_depth but depth modality not loaded.")

        
        # Get action name for sample ID
        action_name = self.metadata.action_names.get(action_id, f"Action_{action_id}")


        sample = {
            "gt_keypoints": gt_keypoints,
            "sample_id": f"{data_info['subject']}_{action_name}_{data_info['subaction']}_{data_info['camera']}_{data_info['start_frame']}",
            "modalities": list(self.modality_names),
            "anchor_key": self.anchor_key,  # Use the actual anchor key
        }
        if "rgb" in self.modality_names:
            sample["input_rgb"] = rgb_frames
            sample["rgb_camera"] = rgb_camera_dict
        if "depth" in self.modality_names:
            sample["input_depth"] = depth_frames
            sample["depth_camera"] = depth_camera_dict

        sample = self.pipeline(sample)

        return sample
