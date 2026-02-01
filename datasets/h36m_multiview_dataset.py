import os
import os.path as osp
import cv2
import numpy as np
import json
import glob
import cdflib
import warnings
from typing import List, Sequence

from datasets.base_dataset import BaseDataset
from .h36m_metadata import load_h36m_metadata


class H36MMultiViewDataset(BaseDataset):
    def __init__(
        self,
        data_root: str = "/data/shared/H36M-Toolbox",
        unit: str = "m",
        pipeline: List[dict] = [],
        split: str = "train",
        cameras: Sequence[str] = ("01", "02", "03", "04"),
        seq_len: int = 5,
        seq_step: int = 1,
        pad_seq: bool = False,
        causal: bool = False,
        remove_static_joints: bool = True,
        return_keypoints_sequence: bool = True,
        resize_hw: Sequence[int] | None = None,
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
        self.cameras = list(cameras)
        self.return_keypoints_sequence = return_keypoints_sequence
        self.resize_hw = tuple(resize_hw) if resize_hw is not None else None

        if unit not in {"mm", "m"}:
            warnings.warn(f"Invalid unit: {unit}. Defaulting to 'm'.")
            self.unit = "m"

        self.images_root = f"{data_root}/images"
        self.extracted_root = f"{data_root}/extracted"

        if split == "train":
            self.subjects = ["S1", "S5", "S6", "S7", "S8"]
        elif split == "test":
            self.subjects = ["S9", "S11"]
        elif split == "train_mini":
            self.subjects = ["S1"]
        elif split == "test_mini":
            self.subjects = ["S11"]
        else:
            self.subjects = ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]

        self.metadata = load_h36m_metadata(f"{data_root}/metadata.xml")
        self.data_list = self._build_dataset()
        self.camera_params = self._load_camera_params()

    def _build_dataset(self):
        data_list = []
        for subject in self.subjects:
            subject_dirs = glob.glob(
                osp.join(self.images_root, f"s_{subject[1:].zfill(2)}_act_*")
            )

            seq_map = {}
            for seq_dir in subject_dirs:
                seq_name = osp.basename(seq_dir)
                parts = seq_name.split("_")
                if len(parts) < 7:
                    continue
                subject_id = parts[1]
                action_id = parts[3]
                subaction_id = parts[5]
                camera_id = parts[7] if len(parts) > 7 else parts[6]

                if camera_id not in self.cameras:
                    continue

                if (subject == "S5" and action_id == "04") or (subject == "S7" and action_id == "15"):
                    continue

                key = (subject_id, action_id, subaction_id)
                seq_map.setdefault(key, {})[camera_id] = seq_dir

            for (subject_id, action_id, subaction_id), cam_dirs in seq_map.items():
                if any(cam not in cam_dirs for cam in self.cameras):
                    continue

                frame_counts = []
                for cam in self.cameras:
                    frame_files = glob.glob(osp.join(cam_dirs[cam], "*.jpg"))
                    frame_counts.append(len(frame_files))
                num_frames = min(frame_counts) if frame_counts else 0

                if num_frames < self.seq_len:
                    continue

                for start_idx in range(0, num_frames - self.seq_len + 1, self.seq_step):
                    data_list.append({
                        "subject_id": int(subject_id),
                        "subject": f"S{int(subject_id)}",
                        "action": action_id,
                        "subaction": subaction_id,
                        "camera_dirs": {cam: cam_dirs[cam] for cam in self.cameras},
                        "start_frame": start_idx,
                        "num_frames": num_frames,
                    })

        return data_list

    def _load_camera_params(self):
        with open(f"{self.h36m_root}/camera-parameters.json", "r") as f:
            json_data = json.load(f)

        camera_params = {}
        camera_id_map = {
            "54138969": 1,
            "55011271": 2,
            "58860488": 3,
            "60457274": 4,
        }
        camera_name_map = {
            "01": "54138969",
            "02": "55011271",
            "03": "58860488",
            "04": "60457274",
        }

        for subject in ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]:
            subject_id = int(subject[1:])
            if subject not in json_data["extrinsics"]:
                continue

            for cam_label in self.cameras:
                cam_name = camera_name_map[cam_label]
                if cam_name not in json_data["extrinsics"][subject]:
                    continue

                extrinsic_data = json_data["extrinsics"][subject][cam_name]
                R = np.array(extrinsic_data["R"], dtype=np.float32)
                t = np.array(extrinsic_data["t"], dtype=np.float32).flatten()

                intrinsic_data = json_data["intrinsics"][cam_name]
                calib_matrix = np.array(intrinsic_data["calibration_matrix"], dtype=np.float32)
                fx, fy = calib_matrix[0, 0], calib_matrix[1, 1]
                cx, cy = calib_matrix[0, 2], calib_matrix[1, 2]
                distortion = np.array(intrinsic_data["distortion"], dtype=np.float32)

                camera_params[(subject_id, cam_label)] = {
                    "R": R,
                    "t": t.reshape(3, 1),
                    "fx": fx,
                    "fy": fy,
                    "cx": cx,
                    "cy": cy,
                    "k": distortion[:3],
                    "p": distortion[3:5],
                    "name": cam_name,
                }

        return camera_params

    def _load_pose_data(self, subject, action, subaction, camera_idx):
        basename = self.metadata.get_base_filename(
            subject,
            "{:d}".format(action),
            "{:d}".format(subaction),
            self.metadata.camera_ids[camera_idx - 1],
        )
        annotname = basename + ".cdf"

        pose_file = osp.join(
            self.extracted_root,
            subject,
            "MyPoseFeatures",
            "D3_Positions_mono_universal",
            annotname,
        )
        if osp.exists(pose_file):
            cdf = cdflib.CDF(pose_file)
            pose_data = np.array(cdf.varget("Pose"))
            if len(pose_data.shape) == 3 and pose_data.shape[2] == 96:
                pose_data = pose_data.transpose(1, 0, 2).reshape(pose_data.shape[1], 32, 3)
            elif len(pose_data.shape) == 3:
                pass
            else:
                pose_data = pose_data.reshape(-1, 32, 3)
            return pose_data

        raise FileNotFoundError(
            f"Pose file not found for subject {subject}, action {action}, subaction {subaction}, camera {camera_idx}"
        )

    def _load_rgb_sequence(self, seq_dir, subject, action, subaction, camera, start_frame):
        frames = []
        for i in range(self.seq_len):
            frame_idx = start_frame + i + 1
            frame_name = (
                f"s_{subject[1:].zfill(2)}_act_{str(action).zfill(2)}_subact_"
                f"{str(subaction).zfill(2)}_ca_{str(camera).zfill(2)}_{frame_idx:06d}.jpg"
            )
            frame_path = osp.join(seq_dir, frame_name)
            if osp.exists(frame_path):
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.resize_hw is not None:
                    frame = cv2.resize(frame, (self.resize_hw[1], self.resize_hw[0]), interpolation=cv2.INTER_LINEAR)
                frames.append(frame)
            else:
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    raise RuntimeError(f"RGB frame not found: {frame_path}")
        return frames

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_info = self.data_list[index]

        rgb_views = []
        gt_keypoints_seq = []
        gt_keypoints_by_view = []
        camera_list = []

        action_id = int(data_info["action"])
        subaction_id = int(data_info["subaction"])

        for cam_label in self.cameras:
            camera_id = int(cam_label)
            seq_dir = data_info["camera_dirs"][cam_label]
            frames = self._load_rgb_sequence(
                seq_dir,
                data_info["subject"],
                action_id,
                subaction_id,
                camera_id,
                data_info["start_frame"],
            )
            if self.resize_hw is None and rgb_views:
                target_h, target_w = rgb_views[0][0].shape[:2]
                resized = []
                for frame in frames:
                    if frame.shape[:2] != (target_h, target_w):
                        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    resized.append(frame)
                frames = resized
            rgb_views.append(frames)

            pose_data = self._load_pose_data(
                data_info["subject"], action_id, subaction_id, camera_id
            )
            pose_start = min(data_info["start_frame"], pose_data.shape[0] - self.seq_len)
            pose_start = max(0, pose_start)
            pose_sequence = pose_data[pose_start : pose_start + self.seq_len]
            if pose_sequence.shape[0] < self.seq_len:
                last_pose = pose_sequence[-1:] if pose_sequence.shape[0] > 0 else np.zeros((1, 32, 3))
                padding_needed = self.seq_len - pose_sequence.shape[0]
                padding = np.repeat(last_pose, padding_needed, axis=0)
                pose_sequence = np.concatenate([pose_sequence, padding], axis=0)

            if self.remove_static_joints:
                joints_to_remove = [4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31]
                pose_sequence = np.delete(pose_sequence, joints_to_remove, axis=1)

            if self.unit == "m":
                pose_sequence = pose_sequence / 1000.0

            gt_keypoints_by_view.append(pose_sequence)

            cam_params = self.camera_params[(data_info["subject_id"], cam_label)]
            T_scaled = cam_params["t"] / 1000.0 if self.unit == "m" else cam_params["t"]
            camera_list.append({
                "intrinsic": np.array(
                    [[cam_params["fx"], 0.0, cam_params["cx"]], [0.0, cam_params["fy"], cam_params["cy"]], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                ),
                "extrinsic": np.hstack((cam_params["R"], T_scaled.reshape(3, 1))).astype(np.float32),
                "camera_id": cam_label,
            })

        try:
            rgb_views = np.stack(rgb_views, axis=0)
        except ValueError as exc:
            shapes = [[frame.shape for frame in view] for view in rgb_views]
            raise ValueError(
                f"Failed to stack RGB views: shape mismatch across cameras. "
                f"cameras={self.cameras}, start_frame={data_info['start_frame']}, "
                f"shapes={shapes}"
            ) from exc
        gt_keypoints_by_view = np.stack(gt_keypoints_by_view, axis=0)

        if self.causal:
            gt_keypoints = gt_keypoints_by_view[:, -1].mean(axis=0)
        else:
            gt_keypoints = gt_keypoints_by_view[:, self.seq_len // 2].mean(axis=0)

        sample = {
            "sample_id": f"{data_info['subject']}_{action_id}_{subaction_id}_{data_info['start_frame']}",
            "modalities": ["rgb"],
            "input_rgb": rgb_views,
            "rgb_cameras": camera_list,
            "gt_keypoints": gt_keypoints,
            "gt_keypoints_by_view": gt_keypoints_by_view if self.return_keypoints_sequence else None,
            "cameras": self.cameras,
        }

        sample = self.pipeline(sample)
        return sample
