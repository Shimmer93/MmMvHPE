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

from h36m_metadata import load_h36m_metadata


class H36MDataset(BaseDataset):
    def __init__(
        self,
        pipeline: List[dict] = [],
        h36m_root: str = "/data/yzhanghe/H36M-Toolbox",
        split: str = "train",
        sequence_length: int = 16,
        frame_step: int = 1,
    ):
        super().__init__(pipeline=pipeline)
        self.h36m_root = h36m_root
        self.split = split
        self.sequence_length = sequence_length
        self.frame_step = frame_step

        # Dataset paths
        self.images_root = f"{h36m_root}/images"
        self.extracted_root = f"{h36m_root}/extracted"

        # H3.6M protocol splits
        if split == "train":
            self.subjects = ["S1", "S5", "S6", "S7", "S8"]
        elif split == "test":
            self.subjects = ["S9", "S11"]
        else:
            self.subjects = ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]

        # Load metadata
        self.metadata = load_h36m_metadata(f"{h36m_root}/metadata.xml")

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

                    # Count available frames
                    frame_files = glob.glob(osp.join(seq_dir, "*.jpg"))
                    num_frames = len(frame_files)

                    if num_frames >= self.sequence_length:
                        # Create sequences with overlap
                        for start_idx in range(
                            0, num_frames - self.sequence_length + 1, self.frame_step
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
        """Load camera parameters for all subjects."""
        with open(f"{self.h36m_root}/camera_data.pkl", "rb") as f:
            camera_params = pickle.load(f)
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

        # Use D3_Positions_mono_universal (world coordinates)
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
            raise FileNotFoundError(
                f"Depth file not found for subject {subject}, action {action}, subaction {subaction}"
            )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_info = self.data_list[index]

        # Load RGB sequence
        rgb_frames = []
        for i in range(self.sequence_length):
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
                data_info["start_frame"], pose_data.shape[0] - self.sequence_length
            )
            pose_start = max(0, pose_start)  # Ensure non-negative
            pose_sequence = pose_data[pose_start : pose_start + self.sequence_length]

            # If we don't have enough frames, pad with the last frame
            if pose_sequence.shape[0] < self.sequence_length:
                last_pose = (
                    pose_sequence[-1:]
                    if pose_sequence.shape[0] > 0
                    else np.zeros((1, 32, 3))
                )
                padding_needed = self.sequence_length - pose_sequence.shape[0]
                padding = np.repeat(last_pose, padding_needed, axis=0)
                pose_sequence = np.concatenate([pose_sequence, padding], axis=0)
        else:
            # Raise error if pose data is missing
            raise RuntimeError(
                f"3D pose data not found for {data_info['subject']} action {action_id} subaction {subaction_id} camera {camera_id}"  
            )

        # Load depth data
        depth_data = self._load_depth_data(data_info['subject'], action_id, subaction_id)
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
        rgb_camera_param = self.camera_params[(data_info['subject_id'], camera_id)]
        rgb_camera_dict = {}
        rgb_camera_dict['R'] = rgb_camera_param[0]
        rgb_camera_dict['T'] = rgb_camera_param[1]
        rgb_camera_dict['fx'] = rgb_camera_param[2][0]
        rgb_camera_dict['fy'] = rgb_camera_param[2][1]
        rgb_camera_dict['cx'] = rgb_camera_param[3][0]
        rgb_camera_dict['cy'] = rgb_camera_param[3][1]
        rgb_camera_dict['k'] = rgb_camera_param[4]
        rgb_camera_dict['p'] = rgb_camera_param[5]

        # Get depth camera parameters (assume same extrinsics as camera 02)
        depth_camera_param = self.camera_params[(data_info['subject_id'], 2)]
        depth_camera_dict = {}
        depth_camera_dict['R'] = depth_camera_param[0]
        depth_camera_dict['T'] = depth_camera_param[1]

        # Get action name for sample ID
        action_name = self.metadata.action_names.get(action_id, f"Action_{action_id}")

        sample = {
            "input_rgb": rgb_frames,
            "input_depth": depth_frames,
            "input_rgb_camera": rgb_camera_dict,
            "input_depth_camera": depth_camera_dict,  # Assuming same camera extrinsics as cam 02
            "gt_keypoints": pose_sequence,
            "sample_id": f"{data_info['subject']}_{action_name}_{data_info['subaction']}_{data_info['camera']}_{data_info['start_frame']}",
            "modalities": ["rgb", "depth"],
            "anchor_key": "input_rgb",  # Coordinates in RGB camera space
        }

        sample = self.pipeline(sample)

        return sample

        # TODO: Check coorespondence between rgb, depth, camera params, and 3D pose data
        # TODO: Check depth camera pose.
