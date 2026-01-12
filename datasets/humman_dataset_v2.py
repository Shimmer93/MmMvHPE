# The dataset assumes each sample captures one and only one human subject.
# The dataset should transform all 3D keypoints, SMPL parameters, and camera extrinsic to a new world coordinate system.
# The new world coordinate system is defined such that the origin is at the pelvis joint of the human model, and the axes are defined such that the smpl model global orientation is (0, 0, 0).
# each sample contains:
# sample_id: str # unique identifier for the sample
# modalities: list # list of available modalities for the sample, e.g., ['rgb', 'depth']
# gt_keypoints: array of shape (24, 3) # ground truth 3D keypoints, centered at the root joint (pelvis)
# gt_smpl_params: array of shape (72+10) # ground truth SMPL parameters (pose and shape), the first 3 values of pose are set to zero (fixed root orientation)
# rgb_camera (if 'rgb' in modalities):
#     intrinsic: array of shape (3, 3) # camera intrinsic matrix
#     extrinsic: array of shape (3, 4) # camera extrinsic matrix (rotation and translation) (may need to be be integrated with translation from raw smpl file)
# depth_camera (if 'depth' in modalities):
#     intrinsic: array of shape (3, 3) # camera intrinsic matrix
#     extrinsic: array of shape (3, 4) # camera extrinsic matrix (rotation and translation) (may need to be be integrated with translation from raw smpl file)
# input_rgb (if 'rgb' in modalities): list/array of shape (seq_len, H, W, 3) # RGB sequence
# input_depth (if 'depth' in modalities): list/array of shape (seq_len, H, W) # depth sequence
# note: anchor_key is not included in this version of the dataset

import os
import os.path as osp
import cv2
import json
import random
import re
import warnings
from typing import List, Optional, Sequence

import numpy as np

from datasets.base_dataset import BaseDataset


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


class HummanPreprocessedDatasetV2(BaseDataset):
    def __init__(
        self,
        data_root: str,
        unit: str = "m",
        pipeline: List[dict] = [],
        split: str = "train",
        modality_names: Sequence[str] = ("rgb", "depth"),
        rgb_cameras: Optional[Sequence[str]] = None,
        depth_cameras: Optional[Sequence[str]] = None,
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
        self.unit = unit
        self.split = split
        self.modality_names = list(modality_names)
        self.seq_len = seq_len
        self.seq_step = seq_step
        self.pad_seq = pad_seq
        self.causal = causal
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

        if unit not in {"mm", "m"}:
            warnings.warn(f"Invalid unit: {unit}. Defaulting to 'm'.")
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
            valid_persons = set(person_ids[split_idx:split_idx + 4])
        else:
            valid_persons = set(person_ids)

        for seq_name in seq_names:
            person_id = seq_name.split("_")[0]
            if person_id not in valid_persons:
                continue

            rgb_cams = list(self.file_index.get("rgb", {}).get(seq_name, {}).keys())
            depth_cams = list(self.file_index.get("depth", {}).get(seq_name, {}).keys())
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

    @staticmethod
    def _flatten_pose(global_orient, body_pose):
        global_orient = np.asarray(global_orient, dtype=np.float32).reshape(-1)
        body_pose = np.asarray(body_pose, dtype=np.float32)
        body_pose = body_pose.reshape(-1)
        return np.concatenate([global_orient, body_pose], axis=0)

    @staticmethod
    def _extract_pelvis(gt_keypoints, gt_transl):
        if gt_keypoints is not None:
            return np.asarray(gt_keypoints[0], dtype=np.float32)
        if gt_transl is not None:
            return np.asarray(gt_transl, dtype=np.float32).reshape(3)
        return np.zeros(3, dtype=np.float32)

    def _to_new_world(self, global_orient, pelvis, points):
        R_root = axis_angle_to_matrix_np(global_orient)
        return (R_root.T @ (points - pelvis).T).T

    def _update_extrinsic(self, R_wc, T_wc, R_root, pelvis):
        R_new = R_wc @ R_root
        T_new = R_wc @ pelvis.reshape(3, 1) + T_wc
        return R_new, T_new

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_info = self.data_list[index].copy()

        if not self.use_all_pairs:
            if data_info.get("rgb_camera") is None and "rgb" in self.modality_names:
                data_info["rgb_camera"] = random.choice(data_info["rgb_cameras"])
            if data_info.get("depth_camera") is None and "depth" in self.modality_names:
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

        gt_global_orient = smpl_params["global_orient"][gt_frame_idx]
        gt_body_pose = smpl_params["body_pose"][gt_frame_idx]
        gt_betas = smpl_params["betas"][gt_frame_idx]
        gt_transl = smpl_params["transl"][gt_frame_idx]
        gt_keypoints = keypoints_3d[gt_frame_idx] if keypoints_3d is not None else None

        pelvis = self._extract_pelvis(gt_keypoints, gt_transl)
        R_root = axis_angle_to_matrix_np(np.asarray(gt_global_orient, dtype=np.float32))

        if gt_keypoints is not None:
            gt_keypoints = self._to_new_world(gt_global_orient, pelvis, gt_keypoints)

        pose = self._flatten_pose(gt_global_orient, gt_body_pose)
        if pose.shape[0] >= 3:
            pose[:3] = 0.0
        pose = pose[:72]
        betas = np.asarray(gt_betas, dtype=np.float32)[:10]
        gt_smpl_params = np.concatenate([pose, betas], axis=0)

        sample = {
            "sample_id": (
                f"{data_info['seq_name']}_rgb_{data_info.get('rgb_camera')}_"
                f"depth_{data_info.get('depth_camera')}_{data_info['start_frame']}"
            ),
            "modalities": list(self.modality_names),
            "gt_keypoints": gt_keypoints,
            "gt_smpl_params": gt_smpl_params,
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
                R_new, T_new = self._update_extrinsic(R, T, R_root, pelvis)
                sample["rgb_camera"] = {
                    "intrinsic": K,
                    "extrinsic": np.hstack((R_new, T_new)).astype(np.float32),
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
                R_new, T_new = self._update_extrinsic(R, T, R_root, pelvis)
                sample["depth_camera"] = {
                    "intrinsic": K,
                    "extrinsic": np.hstack((R_new, T_new)).astype(np.float32),
                }

        sample = self.pipeline(sample)
        return sample
