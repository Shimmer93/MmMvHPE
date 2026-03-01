import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from datasets.humman_dataset_v2 import HummanPreprocessedDatasetV2, axis_angle_to_matrix_np


class HummanPreprocessedDatasetV3(HummanPreprocessedDatasetV2):
    """V2 dataset with optional JSON-loaded RGB/LiDAR skeletons.

    When `rgb_skeleton_json` is provided, this class loads 2D keypoints and
    writes them to `gt_keypoints_2d_rgb` (shape: [T, J, 2]), normalized to
    [-1, 1] in x/y.

    When `lidar_skeleton_json` is provided, this class loads 3D keypoints and
    writes them to `lidar_skeleton_key` (default: `gt_keypoints_lidar`).
    The loaded LiDAR keypoints are converted to match dataset coordinate mode:
    - input can be `"new_world"`, `"world"`, or `"lidar"` (`"camera"` alias)
    - if `apply_to_new_world=True`, output keypoints are in pelvis-centered new-world
      coordinates (with optional root-rotation removal controlled by
      `remove_root_rotation`)
    - else output keypoints are in world coordinates
    """

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
        convert_depth_to_lidar: bool = True,
        apply_to_new_world: bool = True,
        remove_root_rotation: bool = True,
        skeleton_only: bool = True,
        return_keypoints_sequence: bool = False,
        return_smpl_sequence: bool = False,
        rgb_skeleton_json: Optional[str] = None,
        rgb_skeleton_image_size_hw: Sequence[int] = (512, 512),
        lidar_skeleton_json: Optional[str] = None,
        lidar_skeleton_coord: str = "new_world",
        lidar_skeleton_key: str = "gt_keypoints_lidar",
    ):
        super().__init__(
            data_root=data_root,
            unit=unit,
            pipeline=pipeline,
            split=split,
            split_config=split_config,
            split_to_use=split_to_use,
            test_mode=test_mode,
            modality_names=modality_names,
            rgb_cameras=rgb_cameras,
            depth_cameras=depth_cameras,
            rgb_cameras_per_sample=rgb_cameras_per_sample,
            depth_cameras_per_sample=depth_cameras_per_sample,
            lidar_cameras_per_sample=lidar_cameras_per_sample,
            seq_len=seq_len,
            seq_step=seq_step,
            pad_seq=pad_seq,
            causal=causal,
            use_all_pairs=use_all_pairs,
            max_samples=max_samples,
            colocated=colocated,
            convert_depth_to_lidar=convert_depth_to_lidar,
            apply_to_new_world=apply_to_new_world,
            remove_root_rotation=remove_root_rotation,
            skeleton_only=skeleton_only,
            return_keypoints_sequence=return_keypoints_sequence,
            return_smpl_sequence=return_smpl_sequence,
        )

        self.rgb_skeleton_json = rgb_skeleton_json
        self.rgb_skeleton_image_size_hw = (
            int(rgb_skeleton_image_size_hw[0]),
            int(rgb_skeleton_image_size_hw[1]),
        )
        self.rgb_skeleton_index: Dict[str, Dict[str, List[Tuple[int, np.ndarray]]]] = {}
        self.rgb_skeleton_shape: Tuple[int, int] = (17, 2)
        self.lidar_skeleton_json = lidar_skeleton_json
        self.lidar_skeleton_coord = str(lidar_skeleton_coord).lower()
        if self.lidar_skeleton_coord == "camera":
            self.lidar_skeleton_coord = "lidar"
        self.lidar_skeleton_key = str(lidar_skeleton_key)
        self.lidar_skeleton_index: Dict[str, Dict[str, List[Tuple[int, np.ndarray]]]] = {}
        self.lidar_skeleton_shape: Tuple[int, int] = (24, 3)
        if self.lidar_skeleton_coord not in {"new_world", "world", "lidar"}:
            raise ValueError(
                f"Unsupported lidar_skeleton_coord={lidar_skeleton_coord}. "
                "Expected one of {'new_world', 'world', 'lidar'}."
            )

        self._sample_id_re = re.compile(
            r"^(?P<seq>p\d+_a\d+)_rgb_(?P<rgb>kinect_\d{3}|iphone|None)_depth_"
            r"(?P<depth>kinect_\d{3}|iphone|None)_(?P<start>\d+)$"
        )

        if self.rgb_skeleton_json is not None:
            self.rgb_skeleton_index, shape = self._build_rgb_skeleton_index(self.rgb_skeleton_json)
            if shape is not None:
                self.rgb_skeleton_shape = shape
            if not self.rgb_skeleton_index:
                warnings.warn(
                    f"No valid 2D RGB skeleton entries found in {self.rgb_skeleton_json}. "
                    "Falling back to zeros."
                )
        if self.lidar_skeleton_json is not None:
            self.lidar_skeleton_index, shape = self._build_lidar_skeleton_index(self.lidar_skeleton_json)
            if shape is not None:
                self.lidar_skeleton_shape = shape
            if not self.lidar_skeleton_index:
                warnings.warn(
                    f"No valid 3D LiDAR skeleton entries found in {self.lidar_skeleton_json}. "
                    "Falling back to zeros."
                )

    def _build_rgb_skeleton_index(
        self, json_path: str
    ) -> Tuple[Dict[str, Dict[str, List[Tuple[int, np.ndarray]]]], Optional[Tuple[int, int]]]:
        path = Path(json_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict) and "predictions" in payload:
            entries = payload["predictions"]
        elif isinstance(payload, list):
            entries = payload
        elif isinstance(payload, dict):
            entries = [{"image_path": k, "keypoints": v} for k, v in payload.items()]
        else:
            raise ValueError(f"Unsupported JSON format for rgb_skeleton_json: {path}")

        index: Dict[str, Dict[str, List[Tuple[int, np.ndarray]]]] = {}
        inferred_shape: Optional[Tuple[int, int]] = None

        for item in entries:
            if not isinstance(item, dict):
                continue

            rel_path = (
                item.get("image_path")
                or item.get("path")
                or item.get("img_path")
                or item.get("file_name")
            )
            if rel_path is None:
                continue

            keypoints = self._extract_2d_keypoints(item)
            if keypoints is None:
                continue

            seq_name, camera, frame_idx = self._parse_stem_to_index_triplet(Path(rel_path).stem, modality="rgb")
            if seq_name is None or camera is None or frame_idx is None:
                continue

            index.setdefault(seq_name, {}).setdefault(camera, []).append((frame_idx, keypoints))

            if inferred_shape is None:
                inferred_shape = (int(keypoints.shape[0]), int(keypoints.shape[1]))

        for seq_name in index:
            for camera in index[seq_name]:
                index[seq_name][camera].sort(key=lambda x: x[0])

        return index, inferred_shape

    @staticmethod
    def _extract_2d_keypoints(item: Dict) -> Optional[np.ndarray]:
        keypoints = item.get("keypoints")
        if keypoints is None:
            instances = item.get("instances", [])
            if instances:
                if len(instances) == 1:
                    keypoints = instances[0].get("keypoints")
                else:
                    best = max(
                        instances,
                        key=lambda x: float(x.get("bbox_score", 0.0)) if isinstance(x, dict) else 0.0,
                    )
                    if isinstance(best, dict):
                        keypoints = best.get("keypoints")

        if keypoints is None:
            return None

        kpts = np.asarray(keypoints, dtype=np.float32)
        if kpts.ndim == 3 and kpts.shape[0] == 1:
            kpts = kpts[0]
        if kpts.ndim != 2 or kpts.shape[1] < 2:
            return None

        kpts = kpts[:, :2].astype(np.float32)
        if not np.isfinite(kpts).all():
            return None
        return kpts

    def _build_lidar_skeleton_index(
        self, json_path: str
    ) -> Tuple[Dict[str, Dict[str, List[Tuple[int, np.ndarray]]]], Optional[Tuple[int, int]]]:
        path = Path(json_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict) and "predictions" in payload:
            entries = payload["predictions"]
        elif isinstance(payload, list):
            entries = payload
        elif isinstance(payload, dict):
            entries = [{"image_path": k, "keypoints": v} for k, v in payload.items()]
        else:
            raise ValueError(f"Unsupported JSON format for lidar_skeleton_json: {path}")

        index: Dict[str, Dict[str, List[Tuple[int, np.ndarray]]]] = {}
        inferred_shape: Optional[Tuple[int, int]] = None

        for item in entries:
            if not isinstance(item, dict):
                continue

            rel_path = (
                item.get("image_path")
                or item.get("path")
                or item.get("img_path")
                or item.get("file_name")
            )
            if rel_path is None:
                continue

            keypoints = self._extract_3d_keypoints(item)
            if keypoints is None:
                continue

            seq_name, camera, frame_idx = self._parse_stem_to_index_triplet(Path(rel_path).stem, modality="lidar")
            if seq_name is None or camera is None or frame_idx is None:
                continue

            index.setdefault(seq_name, {}).setdefault(camera, []).append((frame_idx, keypoints))

            if inferred_shape is None:
                inferred_shape = (int(keypoints.shape[0]), int(keypoints.shape[1]))

        for seq_name in index:
            for camera in index[seq_name]:
                index[seq_name][camera].sort(key=lambda x: x[0])

        return index, inferred_shape

    @staticmethod
    def _extract_3d_keypoints(item: Dict) -> Optional[np.ndarray]:
        keypoints = item.get("keypoints")
        if keypoints is None:
            instances = item.get("instances", [])
            if instances:
                if len(instances) == 1:
                    keypoints = instances[0].get("keypoints")
                else:
                    best = max(
                        instances,
                        key=lambda x: float(x.get("bbox_score", 0.0)) if isinstance(x, dict) else 0.0,
                    )
                    if isinstance(best, dict):
                        keypoints = best.get("keypoints")

        if keypoints is None:
            return None

        kpts = np.asarray(keypoints, dtype=np.float32)
        if kpts.ndim == 3 and kpts.shape[0] == 1:
            kpts = kpts[0]
        if kpts.ndim != 2:
            return None

        if kpts.shape[1] >= 3:
            kpts = kpts[:, :3].astype(np.float32)
        elif kpts.shape[1] == 2:
            z = np.zeros((kpts.shape[0], 1), dtype=np.float32)
            kpts = np.concatenate([kpts.astype(np.float32), z], axis=1)
        else:
            return None

        if not np.isfinite(kpts).all():
            return None
        return kpts

    def _parse_sample_id(self, sample_id: str) -> Tuple[Optional[str], Optional[str], Optional[str], int]:
        m = self._sample_id_re.match(sample_id)
        if m is None:
            return None, None, None, 0
        seq_name = m.group("seq")
        rgb_camera = None if m.group("rgb") == "None" else m.group("rgb")
        depth_camera = None if m.group("depth") == "None" else m.group("depth")
        start_frame = int(m.group("start"))
        return seq_name, rgb_camera, depth_camera, start_frame

    def _parse_stem_to_index_triplet(
        self, stem: str, modality: str
    ) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        seq_match = self._seq_re.search(stem)
        cam_match = self._cam_re.search(stem)
        frame_match = self._frame_re.search(stem)
        if seq_match and cam_match and frame_match:
            return seq_match.group(1), cam_match.group(1), int(frame_match.group(1))

        seq_name, rgb_camera, depth_camera, start_frame = self._parse_sample_id(stem)
        if seq_name is None:
            return None, None, None
        if modality == "rgb":
            camera = rgb_camera
        elif modality == "lidar":
            camera = depth_camera
        else:
            raise ValueError(f"Unsupported modality `{modality}`.")
        if camera is None:
            camera = "unknown"
        return seq_name, camera, int(start_frame)

    def _resolve_image_size_hw(self, sample) -> Tuple[int, int]:
        rgb = sample.get("input_rgb")
        if isinstance(rgb, torch.Tensor):
            if rgb.dim() >= 4:
                return int(rgb.shape[-2]), int(rgb.shape[-1])
        elif isinstance(rgb, np.ndarray):
            if rgb.ndim == 5:
                return int(rgb.shape[2]), int(rgb.shape[3])
            if rgb.ndim == 4:
                return int(rgb.shape[1]), int(rgb.shape[2])
            if rgb.ndim == 3:
                return int(rgb.shape[0]), int(rgb.shape[1])
        elif isinstance(rgb, (list, tuple)) and len(rgb) > 0:
            frame0 = rgb[0]
            if isinstance(frame0, (list, tuple)) and len(frame0) > 0:
                frame0 = frame0[0]
            if hasattr(frame0, "shape") and len(frame0.shape) >= 2:
                return int(frame0.shape[0]), int(frame0.shape[1])
        return self.rgb_skeleton_image_size_hw

    @staticmethod
    def _normalize_2d(kpts: np.ndarray, image_size_hw: Tuple[int, int]) -> np.ndarray:
        h, w = int(image_size_hw[0]), int(image_size_hw[1])
        if h <= 1 or w <= 1:
            return np.zeros_like(kpts, dtype=np.float32)
        out = kpts.astype(np.float32).copy()
        out[..., 0] = out[..., 0] / (w - 1) * 2.0 - 1.0
        out[..., 1] = out[..., 1] / (h - 1) * 2.0 - 1.0
        return np.clip(out, -1.0, 1.0)

    def _load_rgb_skeleton_frames(
        self, seq_name: str, rgb_camera: Optional[str], start_frame: int
    ) -> np.ndarray:
        zeros = np.zeros(self.rgb_skeleton_shape, dtype=np.float32)
        if rgb_camera is None:
            return np.stack([zeros.copy() for _ in range(self.seq_len)], axis=0)

        seq_data = self.rgb_skeleton_index.get(seq_name, {})
        if not seq_data:
            return np.stack([zeros.copy() for _ in range(self.seq_len)], axis=0)

        if rgb_camera in seq_data:
            frame_list = seq_data[rgb_camera]
        else:
            frame_list = seq_data[sorted(seq_data.keys())[0]]

        if not frame_list:
            return np.stack([zeros.copy() for _ in range(self.seq_len)], axis=0)

        frames = []
        for i in range(self.seq_len):
            idx = min(start_frame + i, len(frame_list) - 1)
            frames.append(frame_list[idx][1].copy())
        return np.stack(frames, axis=0).astype(np.float32)

    def _load_rgb_skeleton_frames_multi(
        self, seq_name: str, rgb_cameras: Sequence[Optional[str]], start_frame: int
    ) -> np.ndarray:
        if not rgb_cameras:
            return self._load_rgb_skeleton_frames(seq_name, None, start_frame)
        view_kpts = []
        for camera_name in rgb_cameras:
            view_kpts.append(self._load_rgb_skeleton_frames(seq_name, camera_name, start_frame))
        if len(view_kpts) == 1:
            return view_kpts[0]
        return np.stack(view_kpts, axis=0).astype(np.float32)

    def _load_lidar_skeleton_frames(
        self, seq_name: str, lidar_camera: Optional[str], start_frame: int
    ) -> np.ndarray:
        zeros = np.zeros(self.lidar_skeleton_shape, dtype=np.float32)
        if lidar_camera is None:
            return np.stack([zeros.copy() for _ in range(self.seq_len)], axis=0)

        seq_data = self.lidar_skeleton_index.get(seq_name, {})
        if not seq_data:
            return np.stack([zeros.copy() for _ in range(self.seq_len)], axis=0)

        if lidar_camera in seq_data:
            frame_list = seq_data[lidar_camera]
        else:
            frame_list = seq_data[sorted(seq_data.keys())[0]]

        if not frame_list:
            return np.stack([zeros.copy() for _ in range(self.seq_len)], axis=0)

        frames = []
        for i in range(self.seq_len):
            idx = min(start_frame + i, len(frame_list) - 1)
            frames.append(frame_list[idx][1].copy())
        return np.stack(frames, axis=0).astype(np.float32)

    def _load_lidar_skeleton_frames_multi(
        self, seq_name: str, lidar_cameras: Sequence[Optional[str]], start_frame: int
    ) -> np.ndarray:
        if not lidar_cameras:
            return self._load_lidar_skeleton_frames(seq_name, None, start_frame)
        view_kpts = []
        for camera_name in lidar_cameras:
            view_kpts.append(self._load_lidar_skeleton_frames(seq_name, camera_name, start_frame))
        if len(view_kpts) == 1:
            return view_kpts[0]
        return np.stack(view_kpts, axis=0).astype(np.float32)

    @staticmethod
    def _to_numpy_3vec(value: Optional[object]) -> np.ndarray:
        if value is None:
            return np.zeros(3, dtype=np.float32)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().astype(np.float32).reshape(3)
        return np.asarray(value, dtype=np.float32).reshape(3)

    @staticmethod
    def _to_numpy_affine(value: Optional[object]) -> np.ndarray:
        if value is None:
            return np.eye(4, dtype=np.float32)
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy().astype(np.float32)
        else:
            arr = np.asarray(value, dtype=np.float32)
        if arr.shape == (4, 4):
            return arr
        if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[1:] == (4, 4):
            return arr[0]
        raise ValueError(f"Expected affine shape (4,4), got {arr.shape}.")

    @staticmethod
    def _to_camera_list(camera_raw):
        if camera_raw is None:
            return []
        if isinstance(camera_raw, (list, tuple)):
            return list(camera_raw)
        return [camera_raw]

    @staticmethod
    def _lidar_to_new_world_single(points: np.ndarray, extrinsic_after: np.ndarray, center: np.ndarray) -> np.ndarray:
        if extrinsic_after.shape != (3, 4):
            raise ValueError(f"Expected lidar extrinsic shape (3,4), got {extrinsic_after.shape}.")
        r = extrinsic_after[:, :3].astype(np.float32)
        t_after = extrinsic_after[:, 3].astype(np.float32)
        t_raw = t_after + center.astype(np.float32)
        pts = points.reshape(-1, 3).astype(np.float32)
        out = (r.T @ (pts - t_raw.reshape(1, 3)).T).T
        return out.reshape(points.shape).astype(np.float32)

    def _transform_lidar_to_new_world(
        self,
        kpts_lidar: np.ndarray,
        lidar_camera,
        input_lidar_affine,
    ) -> np.ndarray:
        camera_list = self._to_camera_list(lidar_camera)
        if len(camera_list) == 0:
            raise ValueError(
                "lidar_skeleton_coord='lidar' requires `lidar_camera` in sample to recover new-world coordinates."
            )
        extrinsics = []
        for cam in camera_list:
            if not isinstance(cam, dict) or "extrinsic" not in cam:
                raise ValueError("Each lidar_camera entry must be a dict containing `extrinsic`.")
            extrinsics.append(np.asarray(cam["extrinsic"], dtype=np.float32))

        affine = self._to_numpy_affine(input_lidar_affine)
        center = -affine[:3, 3].astype(np.float32)
        kpts = np.asarray(kpts_lidar, dtype=np.float32)

        if kpts.ndim == 3:
            return self._lidar_to_new_world_single(kpts, extrinsics[0], center)
        if kpts.ndim == 4:
            num_views = int(kpts.shape[0])
            if len(extrinsics) not in {1, num_views}:
                raise ValueError(
                    f"LiDAR view count mismatch: kpts has {num_views} views but lidar_camera has {len(extrinsics)}."
                )
            out_views = []
            for vidx in range(num_views):
                ext = extrinsics[0] if len(extrinsics) == 1 else extrinsics[vidx]
                out_views.append(self._lidar_to_new_world_single(kpts[vidx], ext, center))
            return np.stack(out_views, axis=0).astype(np.float32)
        raise ValueError(f"Expected LiDAR keypoints with ndim 3 or 4, got shape {kpts.shape}.")

    @staticmethod
    def _transform_new_world_to_world(
        points: np.ndarray,
        global_orient: np.ndarray,
        pelvis: np.ndarray,
        remove_root_rotation: bool = True,
    ) -> np.ndarray:
        if not remove_root_rotation:
            pts = points.reshape(-1, 3)
            pts = pts + pelvis.reshape(1, 3)
            return pts.reshape(points.shape).astype(np.float32)
        r_root = axis_angle_to_matrix_np(np.asarray(global_orient, dtype=np.float32).reshape(3))
        pts = points.reshape(-1, 3)
        pts = (r_root @ pts.T).T + pelvis.reshape(1, 3)
        return pts.reshape(points.shape).astype(np.float32)

    @staticmethod
    def _transform_world_to_new_world(
        points: np.ndarray,
        global_orient: np.ndarray,
        pelvis: np.ndarray,
        remove_root_rotation: bool = True,
    ) -> np.ndarray:
        if not remove_root_rotation:
            pts = points.reshape(-1, 3)
            pts = pts - pelvis.reshape(1, 3)
            return pts.reshape(points.shape).astype(np.float32)
        r_root = axis_angle_to_matrix_np(np.asarray(global_orient, dtype=np.float32).reshape(3))
        pts = points.reshape(-1, 3)
        pts = (r_root.T @ (pts - pelvis.reshape(1, 3)).T).T
        return pts.reshape(points.shape).astype(np.float32)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if not self.rgb_skeleton_index and not self.lidar_skeleton_index:
            return sample

        seq_name = sample.get("seq_name")
        start_frame = sample.get("start_frame")
        selected = sample.get("selected_cameras", {})
        rgb_cameras = selected.get("rgb", []) if isinstance(selected, dict) else []
        lidar_cameras = selected.get("lidar", []) if isinstance(selected, dict) else []

        if seq_name is None or start_frame is None:
            seq_name, rgb_camera, depth_camera, start_frame = self._parse_sample_id(sample.get("sample_id", ""))
            if seq_name is None:
                return sample
            rgb_cameras = [rgb_camera]
            lidar_cameras = [depth_camera]

        if self.rgb_skeleton_index:
            kpts_rgb = self._load_rgb_skeleton_frames_multi(seq_name, rgb_cameras, int(start_frame))
            image_size_hw = self._resolve_image_size_hw(sample)
            kpts_rgb = self._normalize_2d(kpts_rgb, image_size_hw)
            sample["gt_keypoints_2d_rgb"] = torch.from_numpy(kpts_rgb.astype(np.float32))

        if self.lidar_skeleton_index:
            kpts_lidar = self._load_lidar_skeleton_frames_multi(seq_name, lidar_cameras, int(start_frame))
            global_orient = self._to_numpy_3vec(sample.get("gt_global_orient"))
            pelvis = self._to_numpy_3vec(sample.get("gt_pelvis"))

            if self.lidar_skeleton_coord == "lidar":
                kpts_lidar = self._transform_lidar_to_new_world(
                    kpts_lidar,
                    sample.get("lidar_camera"),
                    sample.get("input_lidar_affine"),
                )
                if not self.apply_to_new_world:
                    kpts_lidar = self._transform_new_world_to_world(
                        kpts_lidar,
                        global_orient,
                        pelvis,
                        remove_root_rotation=self.remove_root_rotation,
                    )
            elif self.lidar_skeleton_coord == "new_world" and not self.apply_to_new_world:
                kpts_lidar = self._transform_new_world_to_world(
                    kpts_lidar,
                    global_orient,
                    pelvis,
                    remove_root_rotation=self.remove_root_rotation,
                )
            elif self.lidar_skeleton_coord == "world" and self.apply_to_new_world:
                kpts_lidar = self._transform_world_to_new_world(
                    kpts_lidar,
                    global_orient,
                    pelvis,
                    remove_root_rotation=self.remove_root_rotation,
                )

            sample[self.lidar_skeleton_key] = torch.from_numpy(kpts_lidar.astype(np.float32))
        return sample
