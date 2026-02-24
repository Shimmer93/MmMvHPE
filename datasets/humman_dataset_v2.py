# The dataset assumes each sample captures one and only one human subject.
# The dataset should transform all 3D keypoints, SMPL parameters, and camera extrinsic to a new world coordinate system.
# The new world coordinate system is pelvis-centered.
# If remove_root_rotation=True, the axes remove SMPL global orientation.
# If remove_root_rotation=False, the axes keep world orientation.
# each sample contains:
# sample_id: str # unique identifier for the sample
# modalities: list # list of available modalities for the sample, e.g., ['rgb', 'depth']
# gt_keypoints: array of shape (24, 3) # ground truth 3D keypoints, centered at the root joint (pelvis)
# gt_smpl_params: array of shape (72+10) # ground truth SMPL parameters (pose and shape)
#               # if remove_root_rotation=True, the first 3 values of pose are set to zero
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
import hashlib
import random
import re
import warnings
from typing import List, Optional, Sequence, Dict, Any

import numpy as np
import yaml

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
        split_config: Optional[str] = None,
        split_to_use: str = "random_split",
        test_mode: bool = False,
        modality_names: Sequence[str] = ("rgb", "depth"),
        rgb_cameras: Optional[Sequence[str]] = None,
        depth_cameras: Optional[Sequence[str]] = None,
        rgb_cameras_per_sample: int = 1,
        depth_cameras_per_sample: int = 1,
        lidar_cameras_per_sample: int = 1,
        seq_len: int = 5,
        seq_step: int = 1,
        pad_seq: bool = False,
        causal: bool = False,
        use_all_pairs: bool = False,
        max_samples: Optional[int] = None,
        colocated: bool = False,
        return_keypoints_sequence: bool = False,
        return_smpl_sequence: bool = False,
        convert_depth_to_lidar: bool = True,
        apply_to_new_world: bool = True,
        remove_root_rotation: bool = True,
        skeleton_only: bool = True,
        random_lidar_rotation_deg: float = 0.0,
        random_lidar_rotation_seed: int = 0,
        random_lidar_rotation_ratio: float = 1.0,
    ):
        super().__init__(pipeline=pipeline)
        self.data_root = data_root
        self.unit = unit
        self.split = split
        self.split_config = split_config
        self.split_to_use = split_to_use
        self.test_mode = test_mode
        self.modality_names = list(modality_names)
        self.seq_len = seq_len
        self.seq_step = seq_step
        self.pad_seq = pad_seq
        self.causal = causal
        self.use_all_pairs = use_all_pairs
        self.max_samples = max_samples
        self.colocated = colocated
        self.return_keypoints_sequence = return_keypoints_sequence
        self.return_smpl_sequence = return_smpl_sequence
        self.convert_depth_to_lidar = convert_depth_to_lidar
        self.apply_to_new_world = apply_to_new_world
        self.remove_root_rotation = bool(remove_root_rotation)
        # NOTE: When set to False, RGB/Depth frames are skipped while cameras are still loaded.
        self.skeleton_only = bool(skeleton_only)
        self.random_lidar_rotation_deg = max(0.0, float(random_lidar_rotation_deg))
        self.random_lidar_rotation_seed = int(random_lidar_rotation_seed)
        self.random_lidar_rotation_ratio = float(np.clip(float(random_lidar_rotation_ratio), 0.0, 1.0))


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

        self.lidar_cameras = list(self.depth_cameras)
        self.rgb_cameras_per_sample = max(1, int(rgb_cameras_per_sample))
        self.depth_cameras_per_sample = max(1, int(depth_cameras_per_sample))
        self.lidar_cameras_per_sample = max(1, int(lidar_cameras_per_sample))

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
                indices = random.sample(range(len(self.data_list)), min(self.max_samples, len(self.data_list)))
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

        split_info = self._resolve_split_info(seq_names)
        if split_info is None:
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
            valid_actions = None
            valid_cameras = None
        else:
            valid_persons = set(split_info["subjects"]) if split_info["subjects"] else None
            valid_actions = set(split_info["actions"]) if split_info["actions"] else None
            valid_cameras = set(split_info["cameras"]) if split_info["cameras"] else None

        for seq_name in seq_names:
            person_id = seq_name.split("_")[0]
            if valid_persons is not None and person_id not in valid_persons:
                continue
            action_id = seq_name.split("_")[1]
            if valid_actions is not None and action_id not in valid_actions:
                continue

            rgb_cams = list(self.file_index.get("rgb", {}).get(seq_name, {}).keys())
            depth_cams = list(self.file_index.get("depth", {}).get(seq_name, {}).keys())
            lidar_cams = list(self.file_index.get("lidar", {}).get(seq_name, {}).keys())
            if self.rgb_cameras:
                rgb_cams = [c for c in rgb_cams if c in self.rgb_cameras]
            if self.depth_cameras:
                depth_cams = [c for c in depth_cams if c in self.depth_cameras]
            if self.lidar_cameras:
                lidar_cams = [c for c in lidar_cams if c in self.lidar_cameras]
            if valid_cameras is not None:
                rgb_cams = [c for c in rgb_cams if c in valid_cameras]
                depth_cams = [c for c in depth_cams if c in valid_cameras]
                lidar_cams = [c for c in lidar_cams if c in valid_cameras]

            if "rgb" in self.modality_names and not rgb_cams:
                continue
            if "depth" in self.modality_names and not depth_cams:
                continue
            if "lidar" in self.modality_names and not lidar_cams:
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
                                "lidar_camera": depth_cam if lidar_cams else None,
                                "rgb_cameras": rgb_cams,
                                "depth_cameras": depth_cams,
                                "lidar_cameras": lidar_cams,
                            })
                else:
                    data_list.append({
                        "seq_name": seq_name,
                        "person_id": person_id,
                        "start_frame": start_idx,
                        "num_frames": num_frames,
                        "rgb_camera": None,
                        "depth_camera": None,
                        "lidar_camera": None,
                        "rgb_cameras": rgb_cams,
                        "depth_cameras": depth_cams,
                        "lidar_cameras": lidar_cams,
                    })

        return data_list

    def _resolve_split_info(self, seq_names):
        if self.split_config is None:
            return None

        with open(self.split_config, "r") as f:
            split_config = yaml.safe_load(f)

        if self.split_to_use not in split_config:
            raise ValueError(f"split_to_use {self.split_to_use} not found in {self.split_config}.")

        split_entry = split_config[self.split_to_use]
        split_key = "val_dataset" if self.test_mode else "train_dataset"

        if self.split_to_use == "random_split":
            ratio = split_entry["ratio"]
            seed = split_entry["random_seed"]
            subjects = sorted(list(set([s.split("_")[0] for s in seq_names])))
            rng = np.random.RandomState(seed)
            idx = rng.permutation(len(subjects))
            split_idx = int(np.floor(ratio * len(subjects)))
            train_subjects = [subjects[i] for i in idx[:split_idx]]
            val_subjects = [subjects[i] for i in idx[split_idx:]]
            subjects = train_subjects if split_key == "train_dataset" else val_subjects

            entry = split_entry.get(split_key, {})
            actions = entry.get("actions", None)
            cameras = entry.get("cameras", None)
        else:
            entry = split_entry[split_key]
            subjects = entry.get("subjects", None)
            actions = entry.get("actions", None)
            cameras = entry.get("cameras", None)

        return {"subjects": subjects, "actions": actions, "cameras": cameras}

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

    def _load_lidar_frames(self, seq_name, camera_name, start_frame):
        frame_list = self.file_index["lidar"][seq_name][camera_name]
        frames = []
        for i in range(self.seq_len):
            idx = start_frame + i
            idx = min(idx, len(frame_list) - 1)
            frame_path = frame_list[idx][1]
            pc = np.load(frame_path)
            frames.append(pc.astype(np.float32))
        return frames

    @staticmethod
    def _depth_to_lidar_frames(depth_frames, K, min_depth=1e-6):
        K_inv = np.linalg.inv(K)
        pc_seq = []
        for depth in depth_frames:
            H, W = depth.shape
            xmap, ymap = np.meshgrid(np.arange(W), np.arange(H))
            z = depth.reshape(-1)
            valid = z > min_depth
            pixels = np.stack([xmap.reshape(-1), ymap.reshape(-1), np.ones(H * W)], axis=0)
            rays = K_inv @ pixels
            cam_points = rays * z
            cam_points = cam_points[:, valid]
            pc_seq.append(cam_points.T.astype(np.float32))
        return pc_seq

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

    def _new_world_rotation(self, global_orient):
        if self.remove_root_rotation:
            return axis_angle_to_matrix_np(global_orient)
        return np.eye(3, dtype=np.float32)

    def _to_new_world(self, global_orient, pelvis, points):
        R_new_to_world = self._new_world_rotation(global_orient)
        return (R_new_to_world.T @ (points - pelvis).T).T


    def _update_extrinsic(self, R_wc, T_wc, R_new_to_world, pelvis):
        R_new = R_wc @ R_new_to_world
        T_new = R_wc @ pelvis.reshape(3, 1) + T_wc
        return R_new, T_new

    @staticmethod
    def _rotate_points_np(points: np.ndarray, R_aug: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        out = pts.copy()
        xyz = out[..., :3].reshape(-1, 3)
        out[..., :3] = (xyz @ R_aug.T).reshape(out[..., :3].shape)
        return out

    def _rotate_lidar_sequence(self, lidar_seq, R_aug: np.ndarray):
        if isinstance(lidar_seq, list):
            return [self._rotate_lidar_sequence(x, R_aug) for x in lidar_seq]
        if isinstance(lidar_seq, tuple):
            return tuple(self._rotate_lidar_sequence(x, R_aug) for x in lidar_seq)
        return self._rotate_points_np(lidar_seq, R_aug)

    @staticmethod
    def _rotate_camera_extrinsic(extrinsic: np.ndarray, R_aug: np.ndarray) -> np.ndarray:
        ext = np.asarray(extrinsic, dtype=np.float32)
        if ext.shape != (3, 4):
            raise ValueError(f"Camera extrinsic must be shape (3, 4), got {ext.shape}.")
        R_cam = ext[:, :3]
        T_cam = ext[:, 3:]
        R_rot = R_aug @ R_cam
        T_rot = R_aug @ T_cam
        return np.hstack((R_rot, T_rot)).astype(np.float32)

    def _rotate_camera_container(self, camera_data, R_aug: np.ndarray):
        if camera_data is None:
            return None
        if isinstance(camera_data, list):
            return [self._rotate_camera_container(c, R_aug) for c in camera_data]
        if isinstance(camera_data, tuple):
            return [self._rotate_camera_container(c, R_aug) for c in camera_data]
        if not isinstance(camera_data, dict):
            raise ValueError(f"Camera data must be dict/list/tuple, got {type(camera_data).__name__}.")
        if "extrinsic" not in camera_data:
            raise ValueError("Camera dict missing `extrinsic`.")
        out = dict(camera_data)
        out["extrinsic"] = self._rotate_camera_extrinsic(out["extrinsic"], R_aug)
        return out

    def _sample_sequence_rotation(self, seq_name: str) -> np.ndarray:
        if self.random_lidar_rotation_deg <= 0.0:
            return np.eye(3, dtype=np.float32)
        key = f"rotate:{self.random_lidar_rotation_seed}:{seq_name}".encode("utf-8")
        u64 = int.from_bytes(hashlib.sha1(key).digest()[:8], byteorder="big", signed=False)
        rng = np.random.RandomState(int(u64 % (2 ** 32)))
        angle_deg = float(rng.uniform(-self.random_lidar_rotation_deg, self.random_lidar_rotation_deg))
        axis = rng.normal(size=3).astype(np.float32)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-8:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            axis = axis / axis_norm
        axis_angle = axis * np.deg2rad(angle_deg)
        return axis_angle_to_matrix_np(axis_angle.astype(np.float32))

    def _should_apply_sequence_rotation(self, seq_name: str) -> bool:
        if self.random_lidar_rotation_deg <= 0.0:
            return False
        if self.random_lidar_rotation_ratio >= 1.0:
            return True
        if self.random_lidar_rotation_ratio <= 0.0:
            return False
        key = f"apply:{self.random_lidar_rotation_seed}:{seq_name}".encode("utf-8")
        u64 = int.from_bytes(hashlib.sha1(key).digest()[:8], byteorder="big", signed=False)
        unit = u64 / float((1 << 64) - 1)
        return unit < self.random_lidar_rotation_ratio

    @staticmethod
    def _sample_camera_names(camera_pool: List[str], num_samples: int) -> List[str]:
        if not camera_pool:
            return []
        if len(camera_pool) >= num_samples:
            return random.sample(camera_pool, num_samples)
        sampled = list(camera_pool)
        while len(sampled) < num_samples:
            sampled.append(random.choice(camera_pool))
        return sampled

    @staticmethod
    def _camera_name_to_key(camera_name: str, modality: str) -> str:
        if camera_name.startswith("kinect"):
            suffix = camera_name.split("_")[1]
            if modality == "rgb":
                return f"kinect_color_{suffix}"
            if modality in {"depth", "lidar"}:
                return f"kinect_depth_{suffix}"
        return "iphone"

    @staticmethod
    def _maybe_single(items):
        if len(items) == 1:
            return items[0]
        return items

    @staticmethod
    def _transform_cameras_to_anchor(cameras_list, anchor_R, anchor_T):
        out = []
        inv_anchor_R = np.linalg.inv(anchor_R)
        for cam in cameras_list:
            extrinsic = np.asarray(cam["extrinsic"], dtype=np.float32)
            R_cam = extrinsic[:, :3]
            T_cam = extrinsic[:, 3:]
            R_rel = R_cam @ inv_anchor_R
            T_rel = T_cam - R_rel @ anchor_T
            out.append(
                {
                    "intrinsic": np.asarray(cam["intrinsic"], dtype=np.float32),
                    "extrinsic": np.hstack((R_rel, T_rel)).astype(np.float32),
                }
            )
        return out

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_info = self.data_list[index].copy()

        selected_rgb: List[str] = []
        selected_depth: List[str] = []
        selected_lidar: List[str] = []

        if not self.use_all_pairs:
            if "rgb" in self.modality_names:
                selected_rgb = self._sample_camera_names(
                    list(data_info.get("rgb_cameras", [])),
                    self.rgb_cameras_per_sample,
                )
            if "depth" in self.modality_names:
                selected_depth = self._sample_camera_names(
                    list(data_info.get("depth_cameras", [])),
                    self.depth_cameras_per_sample,
                )
            if "lidar" in self.modality_names:
                if selected_depth:
                    selected_lidar = selected_depth[: self.lidar_cameras_per_sample]
                    if len(selected_lidar) < self.lidar_cameras_per_sample:
                        selected_lidar.extend(
                            self._sample_camera_names(
                                list(data_info.get("lidar_cameras", [])),
                                self.lidar_cameras_per_sample - len(selected_lidar),
                            )
                        )
                else:
                    selected_lidar = self._sample_camera_names(
                        list(data_info.get("lidar_cameras", [])),
                        self.lidar_cameras_per_sample,
                    )
            if self.colocated and "rgb" in self.modality_names and "depth" in self.modality_names:
                common = sorted(
                    list(
                        set(data_info.get("rgb_cameras", []))
                        & set(data_info.get("depth_cameras", []))
                    )
                )
                if common:
                    selected_rgb = self._sample_camera_names(common, self.rgb_cameras_per_sample)
                    selected_depth = self._sample_camera_names(common, self.depth_cameras_per_sample)
                    if "lidar" in self.modality_names:
                        selected_lidar = self._sample_camera_names(common, self.lidar_cameras_per_sample)
        else:
            def _expand_fixed(base_name, pool, count):
                if base_name is None and not pool:
                    return []
                out = [base_name] if base_name is not None else []
                if count <= len(out):
                    return out[:count]
                pool = [p for p in pool if p is not None]
                if base_name is not None:
                    pool = [p for p in pool if p != base_name] + [base_name]
                if not pool:
                    return out
                while len(out) < count:
                    out.append(random.choice(pool))
                return out

            if "rgb" in self.modality_names:
                selected_rgb = _expand_fixed(
                    data_info.get("rgb_camera"),
                    list(data_info.get("rgb_cameras", [])),
                    self.rgb_cameras_per_sample,
                )
            if "depth" in self.modality_names:
                selected_depth = _expand_fixed(
                    data_info.get("depth_camera"),
                    list(data_info.get("depth_cameras", [])),
                    self.depth_cameras_per_sample,
                )
            if "lidar" in self.modality_names:
                selected_lidar = _expand_fixed(
                    data_info.get("lidar_camera"),
                    list(data_info.get("lidar_cameras", [])),
                    self.lidar_cameras_per_sample,
                )

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
        R_new_to_world = self._new_world_rotation(np.asarray(gt_global_orient, dtype=np.float32))

        if gt_keypoints is not None and self.apply_to_new_world:
            gt_keypoints = self._to_new_world(gt_global_orient, pelvis, gt_keypoints)

        pose = self._flatten_pose(gt_global_orient, gt_body_pose)
        if self.remove_root_rotation and pose.shape[0] >= 3:
            pose[:3] = 0.0
        pose = pose[:72]
        betas = np.asarray(gt_betas, dtype=np.float32)[:10]
        gt_smpl_params = np.concatenate([pose, betas], axis=0)

        primary_rgb = selected_rgb[0] if selected_rgb else None
        primary_depth = selected_depth[0] if selected_depth else None
        sample = {
            "sample_id": (
                f"{data_info['seq_name']}_rgb_{primary_rgb}_"
                f"depth_{primary_depth}_{data_info['start_frame']}"
            ),
            "modalities": list(self.modality_names),
            "gt_keypoints": gt_keypoints,
            "gt_smpl_params": gt_smpl_params,
            "gt_global_orient": np.asarray(gt_global_orient, dtype=np.float32),
            "gt_pelvis": np.asarray(pelvis, dtype=np.float32),
            "seq_name": data_info["seq_name"],
            "start_frame": int(data_info["start_frame"]),
            "selected_cameras": {
                "rgb": list(selected_rgb),
                "depth": list(selected_depth),
                "lidar": list(selected_lidar),
            },
        }

        if self.return_keypoints_sequence and keypoints_3d is not None:
            seq_keypoints = []
            for i in range(self.seq_len):
                idx = data_info["start_frame"] + i
                idx = min(idx, keypoints_3d.shape[0] - 1)
                kp = keypoints_3d[idx]
                kp = self._to_new_world(gt_global_orient, pelvis, kp)
                seq_keypoints.append(kp.astype(np.float32))
            sample["gt_keypoints_seq"] = np.stack(seq_keypoints, axis=0)

        if self.return_smpl_sequence:
            seq_smpl = []
            num_smpl_frames = smpl_params["global_orient"].shape[0]
            for i in range(self.seq_len):
                idx = data_info["start_frame"] + i
                idx = min(idx, num_smpl_frames - 1)
                seq_global_orient = smpl_params["global_orient"][idx]
                seq_body_pose = smpl_params["body_pose"][idx]
                seq_betas = smpl_params["betas"][idx]
                seq_pose = self._flatten_pose(seq_global_orient, seq_body_pose)
                if seq_pose.shape[0] >= 3:
                    seq_pose[:3] = 0.0
                seq_pose = seq_pose[:72]
                seq_betas = np.asarray(seq_betas, dtype=np.float32)[:10]
                seq_smpl.append(np.concatenate([seq_pose, seq_betas], axis=0).astype(np.float32))
            sample["gt_smpl_params_seq"] = np.stack(seq_smpl, axis=0)

        if "rgb" in self.modality_names:
            rgb_frames_views = []
            rgb_cameras_out = []
            for camera_name in selected_rgb:
                if self.skeleton_only:
                    rgb_frames = self._load_rgb_frames(
                        data_info["seq_name"],
                        camera_name,
                        data_info["start_frame"],
                    )
                    rgb_frames_views.append(rgb_frames)

                cam_key = self._camera_name_to_key(camera_name, "rgb")
                if cam_key not in cameras:
                    continue
                cam_params = cameras[cam_key]
                K = np.array(cam_params["K"], dtype=np.float32)
                R = np.array(cam_params["R"], dtype=np.float32)
                T = np.array(cam_params["T"], dtype=np.float32).reshape(3, 1)
                R_new, T_new = self._update_extrinsic(R, T, R_new_to_world, pelvis)
                rgb_cameras_out.append(
                    {
                        "intrinsic": K,
                        "extrinsic": np.hstack((R_new, T_new)).astype(np.float32),
                    }
                )

            if self.skeleton_only and rgb_frames_views:
                sample["input_rgb"] = self._maybe_single(rgb_frames_views)
            if rgb_cameras_out:
                sample["rgb_camera"] = self._maybe_single(rgb_cameras_out)

        if "depth" in self.modality_names:
            depth_frames_views = []
            depth_raw_params = []
            depth_cameras_out = []

            for camera_name in selected_depth:
                if self.skeleton_only:
                    depth_frames = self._load_depth_frames(
                        data_info["seq_name"],
                        camera_name,
                        data_info["start_frame"],
                    )
                    depth_frames_views.append(depth_frames)

                cam_key = self._camera_name_to_key(camera_name, "depth")
                if cam_key not in cameras:
                    continue
                cam_params = cameras[cam_key]
                K = np.array(cam_params["K"], dtype=np.float32)
                R = np.array(cam_params["R"], dtype=np.float32)
                T = np.array(cam_params["T"], dtype=np.float32).reshape(3, 1)
                depth_raw_params.append((K, R, T))
                R_new, T_new = self._update_extrinsic(R, T, R_new_to_world, pelvis)
                depth_cameras_out.append(
                    {
                        "intrinsic": K,
                        "extrinsic": np.hstack((R_new, T_new)).astype(np.float32),
                    }
                )

            if self.skeleton_only and depth_frames_views:
                sample["input_depth"] = self._maybe_single(depth_frames_views)

            converted_to_lidar = self.convert_depth_to_lidar and "lidar" not in self.modality_names
            if converted_to_lidar and depth_frames_views:
                if len(depth_frames_views) != len(depth_raw_params):
                    raise ValueError(
                        "Depth-to-lidar conversion requires camera params for every selected depth view."
                    )
                lidar_frames_views = []
                for depth_frames, (K, _, _) in zip(depth_frames_views, depth_raw_params):
                    lidar_frames = self._depth_to_lidar_frames(depth_frames, K)
                    lidar_frames_views.append(lidar_frames)

                if lidar_frames_views:
                    sample["input_lidar"] = self._maybe_single(lidar_frames_views)
                    if "lidar" not in sample["modalities"]:
                        sample["modalities"].append("lidar")
                    if "depth" in sample["modalities"]:
                        sample["modalities"].remove("depth")
                    sample.pop("input_depth", None)
                    sample["selected_cameras"]["lidar"] = list(selected_depth)

                if depth_cameras_out:
                    sample["lidar_camera"] = self._maybe_single(depth_cameras_out)
            elif depth_cameras_out:
                sample["depth_camera"] = self._maybe_single(depth_cameras_out)

        if "lidar" in self.modality_names:
            lidar_frames_views = []
            lidar_cameras_out = []
            for camera_name in selected_lidar:
                lidar_frames = self._load_lidar_frames(
                    data_info["seq_name"],
                    camera_name,
                    data_info["start_frame"],
                )
                lidar_frames_views.append(lidar_frames)

                cam_key = self._camera_name_to_key(camera_name, "lidar")
                if cam_key not in cameras:
                    continue
                cam_params = cameras[cam_key]
                K = np.array(cam_params["K"], dtype=np.float32)
                R = np.array(cam_params["R"], dtype=np.float32)
                T = np.array(cam_params["T"], dtype=np.float32).reshape(3, 1)
                R_new, T_new = self._update_extrinsic(R, T, R_new_to_world, pelvis)
                lidar_cameras_out.append(
                    {
                        "intrinsic": K,
                        "extrinsic": np.hstack((R_new, T_new)).astype(np.float32),
                    }
                )

            if lidar_frames_views:
                sample["input_lidar"] = self._maybe_single(lidar_frames_views)
            if lidar_cameras_out:
                sample["lidar_camera"] = self._maybe_single(lidar_cameras_out)

        if "input_lidar" in sample and self._should_apply_sequence_rotation(data_info["seq_name"]):
            R_aug = self._sample_sequence_rotation(data_info["seq_name"])
            sample["input_lidar"] = self._rotate_lidar_sequence(sample["input_lidar"], R_aug)
            if "lidar_camera" in sample and sample["lidar_camera"] is not None:
                sample["lidar_camera"] = self._rotate_camera_container(sample["lidar_camera"], R_aug)

        sample = self.pipeline(sample)
        return sample
