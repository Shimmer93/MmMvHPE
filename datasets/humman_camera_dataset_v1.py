import os
import os.path as osp
import json
import random
import re
import warnings
from typing import List, Optional, Sequence, Dict, Any

import numpy as np
import torch

from datasets.base_dataset import BaseDataset
from misc.pose_enc import extri_intri_to_pose_encoding


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


def mat_to_quat_np(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float32)
    if quat[3] < 0:
        quat = -quat
    return quat


def quat_to_mat_np(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def slerp_np(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    q1 = q1 / (np.linalg.norm(q1) + 1e-8)
    q2 = q2 / (np.linalg.norm(q2) + 1e-8)
    dot = float(np.dot(q1, q2))
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        q = q1 + t * (q2 - q1)
        return q / (np.linalg.norm(q) + 1e-8)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0 * q1 + s1 * q2).astype(np.float32)


class HummanCameraDatasetV1(BaseDataset):
    """Camera-only training dataset using global 3D keypoints and synthetic cameras."""

    def __init__(
        self,
        data_root: str,
        unit: str = "m",
        pipeline: List[dict] = [],
        split: str = "train",
        split_config: Optional[str] = None,
        split_to_use: str = "random_split",
        test_mode: bool = False,
        modality_names: Sequence[str] = ("rgb", "depth", "lidar"),
        rgb_cameras: Optional[Sequence[str]] = None,
        depth_cameras: Optional[Sequence[str]] = None,
        lidar_cameras: Optional[Sequence[str]] = None,
        mmwave_cameras: Optional[Sequence[str]] = None,
        frame_stride: int = 1,
        max_samples: Optional[int] = None,
        apply_to_new_world: bool = True,
        real_sample_ratio: float = 0.3,
        synthetic_strategy: str = "interpolate",
        synthetic_alpha_range: Sequence[float] = (0.0, 1.0),
        synthetic_rot_jitter_deg: float = 5.0,
        synthetic_trans_jitter: float = 0.05,
        synthetic_focal_scale: Sequence[float] = (0.8, 1.2),
        image_size_hw: Sequence[int] = (224, 224),
        output_pc_centered_lidar: bool = False,
    ):
        super().__init__(pipeline=pipeline)
        self.data_root = data_root
        self.unit = unit
        self.split = split
        self.split_config = split_config
        self.split_to_use = split_to_use
        self.test_mode = test_mode
        self.modality_names = list(modality_names)
        self.frame_stride = max(1, int(frame_stride))
        self.max_samples = max_samples
        self.apply_to_new_world = apply_to_new_world
        self.real_sample_ratio = float(real_sample_ratio)
        self.synthetic_strategy = synthetic_strategy
        self.synthetic_alpha_range = synthetic_alpha_range
        self.synthetic_rot_jitter_deg = float(synthetic_rot_jitter_deg)
        self.synthetic_trans_jitter = float(synthetic_trans_jitter)
        self.synthetic_focal_scale = synthetic_focal_scale
        self.image_size_hw = (int(image_size_hw[0]), int(image_size_hw[1]))
        self.output_pc_centered_lidar = bool(output_pc_centered_lidar)

        self.available_kinect_cameras = [f"kinect_{i:03d}" for i in range(10)]
        self.available_iphone_cameras = ["iphone"]

        self.rgb_cameras = list(rgb_cameras) if rgb_cameras is not None else (
            self.available_kinect_cameras + self.available_iphone_cameras
        )
        self.depth_cameras = list(depth_cameras) if depth_cameras is not None else (
            self.available_kinect_cameras + self.available_iphone_cameras
        )
        self.lidar_cameras = list(lidar_cameras) if lidar_cameras is not None else list(self.depth_cameras)
        self.mmwave_cameras = list(mmwave_cameras) if mmwave_cameras is not None else []

        if unit not in {"mm", "m"}:
            warnings.warn(f"Invalid unit: {unit}. Defaulting to 'm'.")
            self.unit = "m"

        self._seq_re = re.compile(r"(p\d+_a\d+)")

        self.data_list = self._build_dataset()
        if self.max_samples is not None:
            if self.max_samples <= 0:
                self.data_list = []
            else:
                indices = random.sample(range(len(self.data_list)), min(self.max_samples, len(self.data_list)))
                self.data_list = [self.data_list[i] for i in indices]

    def _build_dataset(self):
        data_list = []
        skl_dir = osp.join(self.data_root, "skl")
        if not osp.exists(skl_dir):
            return data_list

        seq_files = [fn for fn in os.listdir(skl_dir) if fn.endswith("_keypoints_3d.npz")]
        seq_names = []
        for fn in seq_files:
            name = fn.replace("_keypoints_3d.npz", "")
            match = self._seq_re.search(name)
            if match:
                seq_names.append(match.group(1))

        seq_names = sorted(set(seq_names))
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
        else:
            valid_persons = set(split_info["subjects"]) if split_info["subjects"] else None
            valid_actions = set(split_info["actions"]) if split_info["actions"] else None

        for seq_name in seq_names:
            person_id = seq_name.split("_")[0]
            if valid_persons is not None and person_id not in valid_persons:
                continue
            action_id = seq_name.split("_")[1]
            if valid_actions is not None and action_id not in valid_actions:
                continue

            keypoints = self._load_keypoints_3d(seq_name)
            if keypoints is None:
                continue
            num_frames = keypoints.shape[0]

            for frame_idx in range(0, num_frames, self.frame_stride):
                data_list.append({
                    "seq_name": seq_name,
                    "frame_idx": frame_idx,
                })

        return data_list

    def _resolve_split_info(self, seq_names):
        if self.split_config is None:
            return None

        import yaml
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
        else:
            entry = split_entry[split_key]
            subjects = entry.get("subjects", None)
            actions = entry.get("actions", None)

        return {"subjects": subjects, "actions": actions}

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

    @staticmethod
    def _update_extrinsic(R_wc, T_wc, R_root, pelvis):
        R_new = R_wc @ R_root
        T_new = R_wc @ pelvis.reshape(3, 1) + T_wc
        return R_new, T_new

    def _build_camera_candidates(self, cameras: Dict[str, Any]):
        candidates = {"rgb": [], "depth": [], "lidar": [], "mmwave": []}

        def add_cam(modality, cam_key):
            if cam_key not in cameras:
                return
            cam_params = cameras[cam_key]
            K = np.array(cam_params["K"], dtype=np.float32)
            R = np.array(cam_params["R"], dtype=np.float32)
            T = np.array(cam_params["T"], dtype=np.float32).reshape(3, 1)
            candidates[modality].append({
                "intrinsic": K,
                "extrinsic": np.hstack((R, T)).astype(np.float32),
            })

        for cam in self.rgb_cameras:
            if cam.startswith("kinect"):
                cam_key = f"kinect_color_{cam.split('_')[1]}"
            else:
                cam_key = "iphone"
            add_cam("rgb", cam_key)

        for cam in self.depth_cameras:
            if cam.startswith("kinect"):
                cam_key = f"kinect_depth_{cam.split('_')[1]}"
            else:
                cam_key = "iphone"
            add_cam("depth", cam_key)

        for cam in self.lidar_cameras:
            if cam.startswith("kinect"):
                cam_key = f"kinect_depth_{cam.split('_')[1]}"
            else:
                cam_key = "iphone"
            add_cam("lidar", cam_key)

        for cam in self.mmwave_cameras:
            add_cam("mmwave", cam)

        return candidates

    def _sample_camera(self, modality: str, candidates: List[Dict[str, np.ndarray]]):
        if candidates:
            base_cam = random.choice(candidates)
        else:
            base_cam = None

        use_real = random.random() < self.real_sample_ratio
        if use_real and base_cam is not None:
            return base_cam

        if self.synthetic_strategy == "interpolate" and len(candidates) >= 2:
            cam_a, cam_b = random.sample(candidates, 2)
            alpha = random.uniform(self.synthetic_alpha_range[0], self.synthetic_alpha_range[1])
            return self._interpolate_cameras(cam_a, cam_b, alpha)

        if base_cam is None:
            return self._random_camera()

        return self._jitter_camera(base_cam)

    def _interpolate_cameras(self, cam_a, cam_b, alpha: float):
        Ra = cam_a["extrinsic"][:, :3]
        Ta = cam_a["extrinsic"][:, 3:]
        Rb = cam_b["extrinsic"][:, :3]
        Tb = cam_b["extrinsic"][:, 3:]

        qa = mat_to_quat_np(Ra)
        qb = mat_to_quat_np(Rb)
        q = slerp_np(qa, qb, alpha)
        R = quat_to_mat_np(q)
        T = (1.0 - alpha) * Ta + alpha * Tb

        K = (1.0 - alpha) * cam_a["intrinsic"] + alpha * cam_b["intrinsic"]
        cam = {"intrinsic": K.astype(np.float32), "extrinsic": np.hstack((R, T)).astype(np.float32)}
        return self._jitter_camera(cam)

    def _jitter_camera(self, cam):
        R = cam["extrinsic"][:, :3]
        T = cam["extrinsic"][:, 3:]

        if self.synthetic_rot_jitter_deg > 0:
            axis = np.random.randn(3).astype(np.float32)
            axis /= np.linalg.norm(axis) + 1e-8
            angle = np.deg2rad(random.uniform(-self.synthetic_rot_jitter_deg, self.synthetic_rot_jitter_deg))
            K = np.array(
                [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
                dtype=np.float32,
            )
            R_jitter = np.eye(3, dtype=np.float32) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
            R = R_jitter @ R

        if self.synthetic_trans_jitter > 0:
            T = T + np.random.randn(3, 1).astype(np.float32) * self.synthetic_trans_jitter

        K = cam["intrinsic"].copy()
        if self.synthetic_focal_scale is not None:
            scale = random.uniform(self.synthetic_focal_scale[0], self.synthetic_focal_scale[1])
            K[0, 0] *= scale
            K[1, 1] *= scale

        return {"intrinsic": K.astype(np.float32), "extrinsic": np.hstack((R, T)).astype(np.float32)}

    def _random_camera(self):
        fx = fy = random.uniform(500.0, 1500.0)
        cx = self.image_size_hw[1] / 2.0
        cy = self.image_size_hw[0] / 2.0
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        axis = np.random.randn(3).astype(np.float32)
        axis /= np.linalg.norm(axis) + 1e-8
        angle = random.uniform(-np.pi, np.pi)
        K_axis = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
            dtype=np.float32,
        )
        R = np.eye(3, dtype=np.float32) + np.sin(angle) * K_axis + (1.0 - np.cos(angle)) * (K_axis @ K_axis)
        T = np.random.randn(3, 1).astype(np.float32)
        return {"intrinsic": K, "extrinsic": np.hstack((R, T)).astype(np.float32)}

    @staticmethod
    def _transform_to_camera(points, extrinsic):
        R = extrinsic[:, :3]
        T = extrinsic[:, 3]
        return (R @ points.T).T + T.reshape(1, 3)

    @staticmethod
    def _project_to_image(points, extrinsic, intrinsic):
        cam_points = HummanCameraDatasetV1._transform_to_camera(points, extrinsic)
        cam_z = np.clip(cam_points[:, 2], 1e-6, None)
        proj = (intrinsic @ cam_points.T).T
        u = proj[:, 0] / cam_z
        v = proj[:, 1] / cam_z
        return np.stack([u, v], axis=-1)

    @staticmethod
    def _normalize_2d(points_2d, image_size_hw):
        height, width = image_size_hw
        x = points_2d[..., 0] / (width - 1) * 2.0 - 1.0
        y = points_2d[..., 1] / (height - 1) * 2.0 - 1.0
        return np.stack([x, y], axis=-1)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_info = self.data_list[index]
        seq_name = data_info["seq_name"]
        frame_idx = data_info["frame_idx"]

        cameras = self._load_camera_params(seq_name)
        smpl_params = self._load_smpl_params(seq_name)
        keypoints_3d = self._load_keypoints_3d(seq_name)
        if keypoints_3d is None:
            raise KeyError(f"Missing keypoints for {seq_name}.")

        gt_frame_idx = min(frame_idx, smpl_params["global_orient"].shape[0] - 1)
        gt_global_orient = smpl_params["global_orient"][gt_frame_idx]
        gt_transl = smpl_params["transl"][gt_frame_idx]
        gt_keypoints = keypoints_3d[gt_frame_idx]

        pelvis = self._extract_pelvis(gt_keypoints, gt_transl)
        R_root = axis_angle_to_matrix_np(np.asarray(gt_global_orient, dtype=np.float32))

        if self.apply_to_new_world:
            gt_keypoints = self._to_new_world(gt_global_orient, pelvis, gt_keypoints)

        sample = {
            "sample_id": f"{seq_name}_{frame_idx}",
            "modalities": list(self.modality_names),
            "gt_keypoints": gt_keypoints.astype(np.float32),
            "gt_global_orient": np.asarray(gt_global_orient, dtype=np.float32),
            "gt_pelvis": np.asarray(pelvis, dtype=np.float32),
            "image_size_hw": self.image_size_hw,
        }

        candidates = self._build_camera_candidates(cameras)
        for modality in self.modality_names:
            cam = self._sample_camera(modality, candidates.get(modality, []))
            if cam is None:
                continue

            extrinsic = cam["extrinsic"]
            intrinsic = cam["intrinsic"]
            if self.apply_to_new_world:
                R_wc = extrinsic[:, :3]
                T_wc = extrinsic[:, 3:]
                R_new, T_new = self._update_extrinsic(R_wc, T_wc, R_root, pelvis)
                extrinsic = np.hstack((R_new, T_new)).astype(np.float32)

            if modality in {"rgb", "depth"}:
                kp_2d = self._project_to_image(gt_keypoints, extrinsic, intrinsic)
                kp_2d = self._normalize_2d(kp_2d, self.image_size_hw)
                kp_2d = np.clip(kp_2d, -1.0, 1.0)
                sample[f"gt_keypoints_2d_{modality}"] = torch.from_numpy(kp_2d.astype(np.float32))
            else:
                kp_3d = self._transform_to_camera(gt_keypoints, extrinsic)
                if modality == "lidar" and self.output_pc_centered_lidar:
                    center = kp_3d.mean(axis=0, keepdims=True)
                    kp_3d = kp_3d - center
                    extrinsic = extrinsic.copy()
                    extrinsic[:, 3:] = extrinsic[:, 3:] - center.T
                    sample["gt_keypoints_pc_centered_input_lidar"] = torch.from_numpy(kp_3d.astype(np.float32))
                    sample["gt_keypoints_pc_center_lidar"] = torch.from_numpy(center.astype(np.float32)).squeeze(0)
                sample[f"gt_keypoints_{modality}"] = torch.from_numpy(kp_3d.astype(np.float32))

            extrinsics = torch.from_numpy(extrinsic.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            intrinsics = torch.from_numpy(intrinsic.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            pose_enc = extri_intri_to_pose_encoding(
                extrinsics,
                intrinsics,
                image_size_hw=self.image_size_hw,
                pose_encoding_type="absT_quaR_FoV",
            )
            sample[f"gt_camera_{modality}"] = pose_enc.squeeze(0)

        sample = self.pipeline(sample)
        return sample