from typing import List, Optional, Sequence

import numpy as np
import torch

from datasets.humman_dataset_v2 import HummanPreprocessedDatasetV2


def build_depth_to_lidar_pipeline(num_points: int = 1024) -> List[dict]:
    num_points = int(num_points)
    if num_points <= 0:
        raise ValueError(f"num_points must be > 0, got {num_points}.")
    return [
        {
            "name": "CameraParamToPoseEncoding",
            "params": {"pose_encoding_type": "absT_quaR_FoV"},
        },
        {
            "name": "PCCenterWithKeypoints",
            "params": {
                "center_type": "mean",
                "keys": ["input_lidar"],
                "keypoints_key": "gt_keypoints",
            },
        },
        {
            "name": "PCPad",
            "params": {
                "num_points": num_points,
                "pad_mode": "repeat",
                "keys": ["input_lidar"],
            },
        },
        {"name": "ToTensor", "params": None},
    ]


class HummanDepthToLidarDataset(HummanPreprocessedDatasetV2):
    """Depth-only HuMMan dataset that uses the V2 depth->LiDAR conversion path."""

    def __init__(
        self,
        data_root: str,
        pipeline: Optional[List[dict]] = None,
        split: str = "train",
        split_config: Optional[str] = None,
        split_to_use: str = "random_split",
        unit: str = "m",
        depth_cameras: Optional[Sequence[str]] = None,
        seq_len: int = 1,
        seq_step: int = 1,
        pad_seq: bool = True,
        causal: bool = True,
        use_all_pairs: bool = False,
        test_mode: bool = False,
        apply_to_new_world: bool = True,
        remove_root_rotation: bool = True,
        colocated: bool = False,
        random_lidar_rotation_deg: float = 0.0,
        random_lidar_rotation_seed: int = 0,
        random_lidar_rotation_ratio: float = 1.0,
        expand_all_depth_cameras: bool = False,
    ) -> None:
        if pipeline is None:
            pipeline = build_depth_to_lidar_pipeline(num_points=1024)

        super().__init__(
            data_root=data_root,
            unit=unit,
            pipeline=pipeline,
            split=split,
            split_config=split_config,
            split_to_use=split_to_use,
            test_mode=test_mode,
            modality_names=("depth",),
            rgb_cameras=(),
            depth_cameras=depth_cameras,
            rgb_cameras_per_sample=1,
            depth_cameras_per_sample=1,
            lidar_cameras_per_sample=1,
            seq_len=seq_len,
            seq_step=seq_step,
            pad_seq=pad_seq,
            causal=causal,
            use_all_pairs=use_all_pairs,
            colocated=colocated,
            convert_depth_to_lidar=True,
            apply_to_new_world=apply_to_new_world,
            remove_root_rotation=remove_root_rotation,
            skeleton_only=True,
        )
        self.random_lidar_rotation_deg = max(0.0, float(random_lidar_rotation_deg))
        self.random_lidar_rotation_seed = int(random_lidar_rotation_seed)
        self.random_lidar_rotation_ratio = float(np.clip(float(random_lidar_rotation_ratio), 0.0, 1.0))
        self.expand_all_depth_cameras = bool(expand_all_depth_cameras)
        if self.expand_all_depth_cameras:
            self._expand_data_list_by_depth_camera()
        # Dedicated RNG so augmentation is controlled by this wrapper only.
        self._rotation_rng = np.random.RandomState(self.random_lidar_rotation_seed)

    def _expand_data_list_by_depth_camera(self) -> None:
        expanded = []
        for data_info in self.data_list:
            depth_cameras = list(data_info.get("depth_cameras", []))
            if len(depth_cameras) == 0:
                expanded.append(data_info)
                continue
            for cam in depth_cameras:
                item = dict(data_info)
                item["depth_camera"] = cam
                item["lidar_camera"] = cam
                item["depth_cameras"] = [cam]
                item["lidar_cameras"] = [cam]
                expanded.append(item)
        self.data_list = expanded

    @staticmethod
    def _axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
        axis_angle = np.asarray(axis_angle, dtype=np.float32)
        angle = float(np.linalg.norm(axis_angle))
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
        I = np.eye(3, dtype=np.float32)
        return I + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

    @staticmethod
    def _rotate_xyz_data(data, rotation: np.ndarray):
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            if data.shape[-1] < 3:
                return data
            out = data.clone()
            rot = torch.as_tensor(rotation, dtype=out.dtype, device=out.device)
            xyz = out[..., :3].reshape(-1, 3)
            out[..., :3] = (xyz @ rot.t()).reshape(out[..., :3].shape)
            return out
        if isinstance(data, np.ndarray):
            if data.shape[-1] < 3:
                return data
            out = np.asarray(data, dtype=np.float32).copy()
            xyz = out[..., :3].reshape(-1, 3)
            out[..., :3] = xyz @ rotation.T
            return out
        if isinstance(data, list):
            return [HummanDepthToLidarDataset._rotate_xyz_data(x, rotation) for x in data]
        if isinstance(data, tuple):
            return [HummanDepthToLidarDataset._rotate_xyz_data(x, rotation) for x in data]
        return data

    @staticmethod
    def _update_affine_rotation(affine, rotation: np.ndarray):
        if affine is None:
            return None
        if isinstance(affine, torch.Tensor):
            if affine.shape != (4, 4):
                return affine
            out = affine.clone()
            rot = torch.as_tensor(rotation, dtype=out.dtype, device=out.device)
            out[:3, :3] = rot @ out[:3, :3]
            return out
        arr = np.asarray(affine, dtype=np.float32)
        if arr.shape != (4, 4):
            return affine
        out = arr.copy()
        out[:3, :3] = rotation @ out[:3, :3]
        return out

    def _sample_random_rotation(self) -> np.ndarray:
        angle_deg = float(
            self._rotation_rng.uniform(-self.random_lidar_rotation_deg, self.random_lidar_rotation_deg)
        )
        axis = self._rotation_rng.normal(size=3).astype(np.float32)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-8:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            axis = axis / axis_norm
        axis_angle = axis * np.deg2rad(angle_deg)
        return self._axis_angle_to_matrix(axis_angle)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if (
            self.test_mode
            or self.random_lidar_rotation_deg <= 0.0
            or self.random_lidar_rotation_ratio <= 0.0
        ):
            return sample
        if self._rotation_rng.rand() >= self.random_lidar_rotation_ratio:
            return sample

        rotation = self._sample_random_rotation()
        sample["input_lidar"] = self._rotate_xyz_data(sample.get("input_lidar", None), rotation)
        if "gt_keypoints_pc_centered_input_lidar" in sample:
            sample["gt_keypoints_pc_centered_input_lidar"] = self._rotate_xyz_data(
                sample.get("gt_keypoints_pc_centered_input_lidar", None), rotation
            )
        elif "gt_keypoints" in sample:
            sample["gt_keypoints"] = self._rotate_xyz_data(sample.get("gt_keypoints", None), rotation)

        if "input_lidar_affine" in sample:
            sample["input_lidar_affine"] = self._update_affine_rotation(
                sample.get("input_lidar_affine", None), rotation
            )

        return sample
