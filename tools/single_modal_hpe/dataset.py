import json
import os
import os.path as osp
import re
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from datasets.base_dataset import BaseDataset


def axis_angle_to_matrix_np(axis_angle: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = axis_angle / angle
    x, y, z = axis
    k = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float32,
    )
    eye = np.eye(3, dtype=np.float32)
    return eye + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


class HummanDepthToLidarDataset(BaseDataset):
    """HuMMan depth-only dataset that outputs depth-derived point clouds and skeletons."""

    def __init__(
        self,
        data_root: str,
        pipeline: List[dict] = [],
        split: str = "train",
        split_config: Optional[str] = None,
        split_to_use: str = "random_split",
        unit: str = "m",
        depth_cameras: Optional[Sequence[str]] = None,
        seq_len: int = 1,
        seq_step: int = 1,
        causal: bool = True,
        test_mode: bool = False,
        apply_to_new_world: bool = True,
        num_points: int = 1024,
        min_depth: float = 1e-6,
    ) -> None:
        super().__init__(pipeline=pipeline)
        self.data_root = data_root
        self.split = split
        self.split_config = split_config
        self.split_to_use = split_to_use
        self.unit = unit
        self.seq_len = max(1, int(seq_len))
        self.seq_step = max(1, int(seq_step))
        self.causal = bool(causal)
        self.test_mode = bool(test_mode)
        self.apply_to_new_world = bool(apply_to_new_world)
        self.num_points = max(1, int(num_points))
        self.min_depth = float(min_depth)

        self.available_cameras = [f"kinect_{i:03d}" for i in range(10)] + ["iphone"]
        self.depth_cameras = list(depth_cameras) if depth_cameras is not None else self.available_cameras

        if self.unit not in {"m", "mm"}:
            raise ValueError("unit must be one of {'m', 'mm'}")

        self._seq_re = re.compile(r"(p\d+_a\d+)")
        self._cam_re = re.compile(r"(kinect_\d{3}|iphone)")
        self._frame_re = re.compile(r"(\d+)$")

        self._camera_cache: Dict[str, Dict] = {}
        self._smpl_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._keypoints_cache: Dict[str, np.ndarray] = {}

        self.file_index = self._index_depth_files()
        self.data_list = self._build_dataset()

    def _index_depth_files(self) -> Dict[str, Dict[str, List[Tuple[int, str]]]]:
        depth_dir = osp.join(self.data_root, "depth")
        if not osp.isdir(depth_dir):
            raise FileNotFoundError(f"Depth folder not found: {depth_dir}")

        file_index: Dict[str, Dict[str, List[Tuple[int, str]]]] = {}
        for file_name in os.listdir(depth_dir):
            base_name, _ = osp.splitext(file_name)
            seq_match = self._seq_re.search(base_name)
            cam_match = self._cam_re.search(base_name)
            frame_match = self._frame_re.search(base_name)
            if not (seq_match and cam_match and frame_match):
                continue

            seq_name = seq_match.group(1)
            camera_name = cam_match.group(1)
            frame_idx = int(frame_match.group(1))
            file_path = osp.join(depth_dir, file_name)
            file_index.setdefault(seq_name, {}).setdefault(camera_name, []).append((frame_idx, file_path))

        for seq_name in file_index:
            for camera_name in file_index[seq_name]:
                file_index[seq_name][camera_name].sort(key=lambda x: x[0])

        return file_index

    def _resolve_split_info(self, seq_names: Sequence[str]) -> Optional[Dict[str, Optional[List[str]]]]:
        if self.split_config is None:
            return None
        if not osp.isfile(self.split_config):
            raise FileNotFoundError(f"Split config not found: {self.split_config}")

        with open(self.split_config, "r", encoding="utf-8") as f:
            split_config = yaml.safe_load(f)

        if self.split_to_use not in split_config:
            raise ValueError(f"split_to_use {self.split_to_use} not found in {self.split_config}")

        split_entry = split_config[self.split_to_use]
        split_key = "val_dataset" if self.test_mode else "train_dataset"

        if self.split_to_use == "random_split":
            ratio = float(split_entry["ratio"])
            seed = int(split_entry["random_seed"])
            subjects = sorted({s.split("_")[0] for s in seq_names})
            rng = np.random.RandomState(seed)
            order = rng.permutation(len(subjects))
            split_idx = int(np.floor(ratio * len(subjects)))
            train_subjects = [subjects[i] for i in order[:split_idx]]
            val_subjects = [subjects[i] for i in order[split_idx:]]
            selected_subjects = train_subjects if split_key == "train_dataset" else val_subjects

            entry = split_entry.get(split_key, {})
            actions = entry.get("actions", None)
            cameras = entry.get("cameras", None)
        else:
            entry = split_entry[split_key]
            selected_subjects = entry.get("subjects", None)
            actions = entry.get("actions", None)
            cameras = entry.get("cameras", None)

        return {
            "subjects": selected_subjects,
            "actions": actions,
            "cameras": cameras,
        }

    def _build_dataset(self) -> List[Dict]:
        data_list: List[Dict] = []
        seq_names = sorted(self.file_index.keys())
        split_info = self._resolve_split_info(seq_names)

        if split_info is None:
            person_ids = sorted({seq.split("_")[0] for seq in seq_names})
            split_idx = int(0.8 * len(person_ids))
            if self.split == "train":
                valid_persons = set(person_ids[:split_idx])
            elif self.split == "test":
                valid_persons = set(person_ids[split_idx:])
            elif self.split == "train_mini":
                valid_persons = set(person_ids[:16])
            elif self.split == "test_mini":
                valid_persons = set(person_ids[split_idx : split_idx + 4])
            else:
                valid_persons = set(person_ids)
            valid_actions = None
            valid_cameras = None
        else:
            valid_persons = set(split_info["subjects"]) if split_info["subjects"] else None
            valid_actions = set(split_info["actions"]) if split_info["actions"] else None
            valid_cameras = set(split_info["cameras"]) if split_info["cameras"] else None

        for seq_name in seq_names:
            person_id, action_id = seq_name.split("_")
            if valid_persons is not None and person_id not in valid_persons:
                continue
            if valid_actions is not None and action_id not in valid_actions:
                continue

            cameras = sorted(self.file_index[seq_name].keys())
            if self.depth_cameras:
                cameras = [cam for cam in cameras if cam in self.depth_cameras]
            if valid_cameras is not None:
                cameras = [cam for cam in cameras if cam in valid_cameras]
            if not cameras:
                continue

            for camera_name in cameras:
                frame_list = self.file_index[seq_name][camera_name]
                num_frames = len(frame_list)
                if num_frames < self.seq_len:
                    continue

                for start_idx in range(0, num_frames - self.seq_len + 1, self.seq_step):
                    data_list.append(
                        {
                            "seq_name": seq_name,
                            "person_id": person_id,
                            "camera_name": camera_name,
                            "start_frame": start_idx,
                        }
                    )

        return data_list

    def _load_camera_params(self, seq_name: str) -> Dict:
        if seq_name not in self._camera_cache:
            camera_file = osp.join(self.data_root, "cameras", f"{seq_name}_cameras.json")
            with open(camera_file, "r", encoding="utf-8") as f:
                self._camera_cache[seq_name] = json.load(f)
        return self._camera_cache[seq_name]

    def _load_smpl_params(self, seq_name: str) -> Dict[str, np.ndarray]:
        if seq_name not in self._smpl_cache:
            smpl_file = osp.join(self.data_root, "smpl", f"{seq_name}_smpl_params.npz")
            smpl_data = np.load(smpl_file)
            self._smpl_cache[seq_name] = {
                "global_orient": smpl_data["global_orient"],
                "body_pose": smpl_data["body_pose"],
                "betas": smpl_data["betas"],
                "transl": smpl_data["transl"],
            }
        return self._smpl_cache[seq_name]

    def _load_keypoints_3d(self, seq_name: str) -> np.ndarray:
        if seq_name not in self._keypoints_cache:
            keypoints_file = osp.join(self.data_root, "skl", f"{seq_name}_keypoints_3d.npz")
            if not osp.exists(keypoints_file):
                raise FileNotFoundError(
                    f"Missing keypoints file for {seq_name}: {keypoints_file}. "
                    "Run tools/generate_humman_smpl_outputs.py first."
                )
            self._keypoints_cache[seq_name] = np.load(keypoints_file)["keypoints_3d"]
        return self._keypoints_cache[seq_name]

    @staticmethod
    def _camera_name_to_key(camera_name: str) -> str:
        if camera_name.startswith("kinect"):
            suffix = camera_name.split("_")[1]
            return f"kinect_depth_{suffix}"
        return "iphone"

    def _load_depth_frames(self, seq_name: str, camera_name: str, start_frame: int) -> List[np.ndarray]:
        frame_list = self.file_index[seq_name][camera_name]
        frames: List[np.ndarray] = []
        for i in range(self.seq_len):
            idx = min(start_frame + i, len(frame_list) - 1)
            depth_path = frame_list[idx][1]
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            if depth is None:
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                if frames:
                    depth = frames[-1].copy()
                else:
                    depth = np.zeros((512, 512), dtype=np.float32)
            depth = depth.astype(np.float32)
            if self.unit == "m":
                depth = depth / 1000.0
            frames.append(depth)
        return frames

    @staticmethod
    def _depth_to_world_points(
        depth: np.ndarray,
        k_inv: np.ndarray,
        r_wc: np.ndarray,
        t_wc: np.ndarray,
        min_depth: float,
    ) -> np.ndarray:
        h, w = depth.shape
        xmap, ymap = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.reshape(-1)
        valid = z > min_depth

        pixels = np.stack([xmap.reshape(-1), ymap.reshape(-1), np.ones(h * w)], axis=0)
        rays = k_inv @ pixels
        cam_points = rays * z
        cam_points = cam_points[:, valid]
        world_points = (r_wc.T @ (cam_points - t_wc)).T
        return world_points.astype(np.float32)

    @staticmethod
    def _extract_pelvis(gt_keypoints: np.ndarray, gt_transl: np.ndarray) -> np.ndarray:
        if gt_keypoints is not None:
            return np.asarray(gt_keypoints[0], dtype=np.float32)
        return np.asarray(gt_transl, dtype=np.float32).reshape(3)

    @staticmethod
    def _to_new_world(points: np.ndarray, r_root: np.ndarray, pelvis: np.ndarray) -> np.ndarray:
        return (r_root.T @ (points - pelvis).T).T.astype(np.float32)

    @staticmethod
    def _update_extrinsic(R_wc, T_wc, R_root, pelvis):
        R_new = R_wc @ R_root
        T_new = R_wc @ pelvis.reshape(3, 1) + T_wc
        return R_new, T_new

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> Dict:
        data_info = self.data_list[index]
        seq_name = data_info["seq_name"]
        camera_name = data_info["camera_name"]
        start_frame = int(data_info["start_frame"])
        frame_list = self.file_index[seq_name][camera_name]

        cameras = self._load_camera_params(seq_name)
        smpl_params = self._load_smpl_params(seq_name)
        keypoints_3d = self._load_keypoints_3d(seq_name)

        if self.causal:
            gt_frame_idx = start_frame + self.seq_len - 1
        else:
            gt_frame_idx = start_frame + self.seq_len // 2
        gt_frame_idx = min(gt_frame_idx, smpl_params["global_orient"].shape[0] - 1)

        gt_global_orient = np.asarray(smpl_params["global_orient"][gt_frame_idx], dtype=np.float32)
        gt_transl = np.asarray(smpl_params["transl"][gt_frame_idx], dtype=np.float32)
        gt_keypoints = np.asarray(keypoints_3d[gt_frame_idx], dtype=np.float32)

        pelvis = self._extract_pelvis(gt_keypoints, gt_transl)
        r_root = axis_angle_to_matrix_np(gt_global_orient)

        if self.apply_to_new_world:
            gt_keypoints = self._to_new_world(gt_keypoints, r_root, pelvis)

        cam_key = self._camera_name_to_key(camera_name)
        if cam_key not in cameras:
            raise KeyError(f"Camera key {cam_key} not found in {seq_name}_cameras.json")
        cam_params = cameras[cam_key]
        k = np.asarray(cam_params["K"], dtype=np.float32)
        r_wc = np.asarray(cam_params["R"], dtype=np.float32)
        t_wc = np.asarray(cam_params["T"], dtype=np.float32).reshape(3, 1)
        k_inv = np.linalg.inv(k)
        if self.apply_to_new_world:
            r_cam, t_cam = self._update_extrinsic(r_wc, t_wc, r_root, pelvis)
        else:
            r_cam, t_cam = r_wc, t_wc

        depth_frames = self._load_depth_frames(seq_name, camera_name, start_frame)
        lidar_seq = []
        for depth in depth_frames:
            world_points = self._depth_to_world_points(depth, k_inv, r_wc, t_wc, self.min_depth)
            if self.apply_to_new_world:
                world_points = self._to_new_world(world_points, r_root, pelvis)
            lidar_seq.append(world_points.astype(np.float32))

        if self.causal:
            frame_pick = start_frame + self.seq_len - 1
        else:
            frame_pick = start_frame + self.seq_len // 2
        frame_pick = min(frame_pick, len(frame_list) - 1)
        frame_path_abs = frame_list[frame_pick][1]
        depth_root = osp.join(self.data_root, "depth")
        frame_path_rel = osp.relpath(frame_path_abs, depth_root).replace("\\", "/")

        sample = {
            "sample_id": f"{seq_name}_{camera_name}_{start_frame:06d}",
            "frame_path": frame_path_rel,
            "modalities": ["lidar"],
            "input_lidar": lidar_seq,
            "lidar_camera": {
                "intrinsic": k.astype(np.float32),
                "extrinsic": np.hstack((r_cam, t_cam)).astype(np.float32),
            },
            "gt_keypoints": gt_keypoints.astype(np.float32),
        }
        sample = self.pipeline(sample)
        return sample
