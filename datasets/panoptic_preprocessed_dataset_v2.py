import bisect
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from datasets.panoptic_preprocessed_dataset_v1 import PanopticPreprocessedDatasetV1


class PanopticPreprocessedDatasetV2(PanopticPreprocessedDatasetV1):
    """V1 dataset with optional JSON-loaded RGB/LiDAR skeletons.

    - `rgb_skeleton_json` -> writes normalized 2D keypoints to `gt_keypoints_2d_rgb`.
    - `lidar_skeleton_json` -> writes 3D keypoints to `lidar_skeleton_key`.
    """

    def __init__(
        self,
        data_root: str = "/opt/data/panoptic_kinoptic_single_actor_cropped",
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
        convert_depth_to_lidar: bool = False,
        skeleton_only: bool = True,
        return_keypoints_sequence: bool = False,
        return_smpl_sequence: bool = False,
        sequence_allowlist: Optional[Sequence[str]] = None,
        sequence_list_file: Optional[str] = None,
        strict_validation: bool = True,
        random_seed: int = 0,
        gt_unit: str = "cm",
        output_num_joints: int = 19,
        apply_to_new_world: bool = False,
        remove_root_rotation: bool = False,
        root_rotation_fallback: str = "skip",
        max_skip_invalid_samples: int = 64,
        panoptic_toolbox_root: Optional[str] = None,
        use_panoptic_calibration_extrinsics: bool = False,
        rgb_skeleton_json: Optional[str] = None,
        rgb_skeleton_image_size_hw: Sequence[int] = (512, 512),
        lidar_skeleton_json: Optional[str] = None,
        lidar_skeleton_coord: str = "new_world",
        lidar_skeleton_key: str = "gt_keypoints_lidar",
        lidar_skeleton_is_pc_centered: bool = False,
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
            skeleton_only=skeleton_only,
            return_keypoints_sequence=return_keypoints_sequence,
            return_smpl_sequence=return_smpl_sequence,
            sequence_allowlist=sequence_allowlist,
            sequence_list_file=sequence_list_file,
            strict_validation=strict_validation,
            random_seed=random_seed,
            gt_unit=gt_unit,
            output_num_joints=output_num_joints,
            apply_to_new_world=apply_to_new_world,
            remove_root_rotation=remove_root_rotation,
            root_rotation_fallback=root_rotation_fallback,
            max_skip_invalid_samples=max_skip_invalid_samples,
            panoptic_toolbox_root=panoptic_toolbox_root,
            use_panoptic_calibration_extrinsics=use_panoptic_calibration_extrinsics,
        )

        self.rgb_skeleton_json = rgb_skeleton_json
        self.rgb_skeleton_image_size_hw = (
            int(rgb_skeleton_image_size_hw[0]),
            int(rgb_skeleton_image_size_hw[1]),
        )
        self.rgb_skeleton_index: Dict[str, Dict[str, List[Tuple[int, np.ndarray]]]] = {}
        self.rgb_skeleton_shape: Tuple[int, int] = (self.output_num_joints, 2)

        self.lidar_skeleton_json = lidar_skeleton_json
        self.lidar_skeleton_coord = str(lidar_skeleton_coord).lower()
        if self.lidar_skeleton_coord == "camera":
            self.lidar_skeleton_coord = "lidar"
        self.lidar_skeleton_key = str(lidar_skeleton_key)
        self.lidar_skeleton_is_pc_centered = bool(lidar_skeleton_is_pc_centered)
        self.lidar_skeleton_index: Dict[str, Dict[str, List[Tuple[int, np.ndarray]]]] = {}
        self.lidar_skeleton_shape: Tuple[int, int] = (self.output_num_joints, 3)
        if self.lidar_skeleton_coord not in {"new_world", "world", "lidar"}:
            raise ValueError(
                f"Unsupported lidar_skeleton_coord={lidar_skeleton_coord}. "
                "Expected one of {'new_world', 'world', 'lidar'}."
            )
        if self.lidar_skeleton_is_pc_centered and self.lidar_skeleton_coord != "lidar":
            raise ValueError(
                "lidar_skeleton_is_pc_centered=True requires lidar_skeleton_coord='lidar'."
            )

        self._frame_re = re.compile(r"(\d+)$")
        self._sample_id_re = re.compile(
            r"^(?P<seq>.+?)_rgb_(?P<rgb>kinect_\d+|iphone|None)_depth_"
            r"(?P<depth>kinect_\d+|iphone|None)_(?P<frame>\d+)$"
        )
        self._cam_re = re.compile(r"(kinect_\d+|iphone)")

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
            seq_name, camera, frame_idx = self._parse_path_to_index_triplet(str(rel_path), modality="rgb")
            if seq_name is None or camera is None or frame_idx is None:
                continue
            index.setdefault(seq_name, {}).setdefault(camera, []).append((frame_idx, keypoints))
            if inferred_shape is None:
                inferred_shape = (int(keypoints.shape[0]), int(keypoints.shape[1]))

        for seq_name in index:
            for camera in index[seq_name]:
                index[seq_name][camera].sort(key=lambda x: x[0])
        return index, inferred_shape

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
            seq_name, camera, frame_idx = self._parse_path_to_index_triplet(str(rel_path), modality="lidar")
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

    def _parse_sample_id(self, sample_id: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[int]]:
        m = self._sample_id_re.match(str(sample_id))
        if m is None:
            return None, None, None, None
        seq_name = m.group("seq")
        rgb_camera = None if m.group("rgb") == "None" else self._normalize_camera_name(m.group("rgb"))
        depth_camera = None if m.group("depth") == "None" else self._normalize_camera_name(m.group("depth"))
        return seq_name, rgb_camera, depth_camera, int(m.group("frame"))

    def _parse_path_to_index_triplet(
        self,
        rel_path: str,
        modality: str,
    ) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        p = Path(rel_path)
        parts = [str(x) for x in p.parts]
        frame_idx = self._extract_frame_idx_from_stem(p.stem)

        modality_dirs = {"rgb"} if modality == "rgb" else {"depth", "lidar"}
        for i, part in enumerate(parts):
            low = part.lower()
            if low not in modality_dirs:
                continue
            if i >= 1 and i + 1 < len(parts):
                seq_name = parts[i - 1]
                camera = self._normalize_camera_name(parts[i + 1])
                if frame_idx is not None:
                    return seq_name, camera, frame_idx

        seq_name, rgb_camera, depth_camera, frame_from_id = self._parse_sample_id(p.stem)
        if seq_name is not None and frame_from_id is not None:
            if modality == "rgb":
                return seq_name, rgb_camera, frame_from_id
            return seq_name, depth_camera, frame_from_id

        cam_match = self._cam_re.search(p.stem)
        if frame_idx is not None and cam_match is not None:
            camera = self._normalize_camera_name(cam_match.group(1))
            for seq_name in self.sequence_data.keys():
                if p.stem.startswith(f"{seq_name}_"):
                    return seq_name, camera, frame_idx
        return None, None, None

    def _extract_frame_idx_from_stem(self, stem: str) -> Optional[int]:
        m = self._frame_re.search(str(stem))
        if m is None:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    @staticmethod
    def _normalize_2d(kpts: np.ndarray, image_size_hw: Tuple[int, int]) -> np.ndarray:
        h, w = int(image_size_hw[0]), int(image_size_hw[1])
        if h <= 1 or w <= 1:
            return np.zeros_like(kpts, dtype=np.float32)
        out = kpts.astype(np.float32).copy()
        out[..., 0] = out[..., 0] / (w - 1) * 2.0 - 1.0
        out[..., 1] = out[..., 1] / (h - 1) * 2.0 - 1.0
        return np.clip(out, -1.0, 1.0)

    def _resolve_image_size_hw(self, sample) -> Tuple[int, int]:
        rgb = sample.get("input_rgb")
        if isinstance(rgb, torch.Tensor):
            if rgb.dim() == 4:
                # [T,C,H,W] or [T,H,W,C]
                if int(rgb.shape[1]) in {1, 3}:
                    return int(rgb.shape[2]), int(rgb.shape[3])
                return int(rgb.shape[1]), int(rgb.shape[2])
            if rgb.dim() >= 5:
                # [V,T,C,H,W] or [V,T,H,W,C]
                if int(rgb.shape[2]) in {1, 3}:
                    return int(rgb.shape[3]), int(rgb.shape[4])
                return int(rgb.shape[2]), int(rgb.shape[3])
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

    def _build_frame_window(self, seq_name: str, start_frame: int) -> List[int]:
        frame_ids = self.sequence_data[seq_name]["frame_ids"]
        start = int(start_frame)
        frame_window: List[int] = []
        for i in range(self.seq_len):
            idx = start + i
            if idx >= len(frame_ids):
                if not self.pad_seq:
                    break
                idx = len(frame_ids) - 1
            frame_window.append(int(frame_ids[idx]))
        if len(frame_window) == 0:
            raise ValueError(f"Empty frame window for seq={seq_name}, start={start_frame}")
        return frame_window

    def _body_frame_to_start_index(self, seq_name: str, body_frame_id: int) -> int:
        frame_ids = self.sequence_data.get(seq_name, {}).get("frame_ids", [])
        if not frame_ids:
            return 0
        frame_ids = [int(x) for x in frame_ids]
        body_frame_id = int(body_frame_id)
        pos = bisect.bisect_left(frame_ids, body_frame_id)
        if pos < len(frame_ids) and frame_ids[pos] == body_frame_id:
            return int(pos)
        if pos == 0:
            return 0
        if pos >= len(frame_ids):
            return len(frame_ids) - 1
        left = frame_ids[pos - 1]
        right = frame_ids[pos]
        return pos - 1 if abs(body_frame_id - left) <= abs(right - body_frame_id) else pos

    @staticmethod
    def _select_frame_by_body_id(frame_list: List[Tuple[int, np.ndarray]], body_frame_id: int) -> np.ndarray:
        if len(frame_list) == 0:
            raise ValueError("Cannot select frame from empty skeleton list.")
        ids = [int(x[0]) for x in frame_list]
        id_to_data = {int(fid): data for fid, data in frame_list}
        key = int(body_frame_id)
        if key in id_to_data:
            return id_to_data[key]
        pos = bisect.bisect_left(ids, key)
        if pos == 0:
            return id_to_data[ids[0]]
        if pos >= len(ids):
            return id_to_data[ids[-1]]
        left = ids[pos - 1]
        right = ids[pos]
        chosen = left if abs(key - left) <= abs(right - key) else right
        return id_to_data[chosen]

    def _load_rgb_skeleton_frames(
        self,
        seq_name: str,
        rgb_camera: Optional[str],
        frame_window: Sequence[int],
    ) -> np.ndarray:
        zeros = np.zeros(self.rgb_skeleton_shape, dtype=np.float32)
        if rgb_camera is None:
            return np.stack([zeros.copy() for _ in frame_window], axis=0)
        seq_data = self.rgb_skeleton_index.get(seq_name, {})
        if not seq_data:
            return np.stack([zeros.copy() for _ in frame_window], axis=0)
        camera = self._normalize_camera_name(rgb_camera)
        frame_list = seq_data.get(camera, None)
        if frame_list is None:
            frame_list = seq_data[sorted(seq_data.keys())[0]]
        frames = [self._select_frame_by_body_id(frame_list, int(fid)).copy() for fid in frame_window]
        return np.stack(frames, axis=0).astype(np.float32)

    def _load_rgb_skeleton_frames_multi(
        self,
        seq_name: str,
        rgb_cameras: Sequence[Optional[str]],
        frame_window: Sequence[int],
    ) -> np.ndarray:
        if not rgb_cameras:
            return self._load_rgb_skeleton_frames(seq_name, None, frame_window)
        view_kpts = []
        for camera_name in rgb_cameras:
            view_kpts.append(self._load_rgb_skeleton_frames(seq_name, camera_name, frame_window))
        if len(view_kpts) == 1:
            return view_kpts[0]
        return np.stack(view_kpts, axis=0).astype(np.float32)

    def _load_lidar_skeleton_frames(
        self,
        seq_name: str,
        lidar_camera: Optional[str],
        frame_window: Sequence[int],
    ) -> np.ndarray:
        zeros = np.zeros(self.lidar_skeleton_shape, dtype=np.float32)
        if lidar_camera is None:
            return np.stack([zeros.copy() for _ in frame_window], axis=0)
        seq_data = self.lidar_skeleton_index.get(seq_name, {})
        if not seq_data:
            return np.stack([zeros.copy() for _ in frame_window], axis=0)
        camera = self._normalize_camera_name(lidar_camera)
        frame_list = seq_data.get(camera, None)
        if frame_list is None:
            frame_list = seq_data[sorted(seq_data.keys())[0]]
        frames = [self._select_frame_by_body_id(frame_list, int(fid)).copy() for fid in frame_window]
        return np.stack(frames, axis=0).astype(np.float32)

    def _load_lidar_skeleton_frames_multi(
        self,
        seq_name: str,
        lidar_cameras: Sequence[Optional[str]],
        frame_window: Sequence[int],
    ) -> np.ndarray:
        if not lidar_cameras:
            return self._load_lidar_skeleton_frames(seq_name, None, frame_window)
        view_kpts = []
        for camera_name in lidar_cameras:
            view_kpts.append(self._load_lidar_skeleton_frames(seq_name, camera_name, frame_window))
        if len(view_kpts) == 1:
            return view_kpts[0]
        return np.stack(view_kpts, axis=0).astype(np.float32)

    @staticmethod
    def _to_camera_list(camera_raw):
        if camera_raw is None:
            return []
        if isinstance(camera_raw, (list, tuple)):
            return list(camera_raw)
        return [camera_raw]

    @staticmethod
    def _lidar_to_target_single(points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
        ext = np.asarray(extrinsic, dtype=np.float32)
        if ext.shape != (3, 4):
            raise ValueError(f"Expected lidar extrinsic shape (3,4), got {ext.shape}.")
        r = ext[:, :3].astype(np.float32)
        t = ext[:, 3].astype(np.float32)
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        out = (r.T @ (pts - t.reshape(1, 3)).T).T
        return out.reshape(points.shape).astype(np.float32)

    def _transform_lidar_to_target_frame(
        self,
        kpts_lidar: np.ndarray,
        lidar_camera,
    ) -> np.ndarray:
        camera_list = self._to_camera_list(lidar_camera)
        if len(camera_list) == 0:
            raise ValueError(
                "lidar_skeleton_coord='lidar' requires `lidar_camera`/`depth_camera` in sample."
            )
        extrinsics = []
        for cam in camera_list:
            if not isinstance(cam, dict) or "extrinsic" not in cam:
                raise ValueError("Each lidar_camera entry must be a dict containing `extrinsic`.")
            extrinsics.append(np.asarray(cam["extrinsic"], dtype=np.float32))

        kpts = np.asarray(kpts_lidar, dtype=np.float32)
        if kpts.ndim == 3:
            return self._lidar_to_target_single(kpts, extrinsics[0])
        if kpts.ndim == 4:
            num_views = int(kpts.shape[0])
            if len(extrinsics) not in {1, num_views}:
                raise ValueError(
                    f"LiDAR view count mismatch: kpts has {num_views} views but camera list has {len(extrinsics)}."
                )
            out_views = []
            for vidx in range(num_views):
                ext = extrinsics[0] if len(extrinsics) == 1 else extrinsics[vidx]
                out_views.append(self._lidar_to_target_single(kpts[vidx], ext))
            return np.stack(out_views, axis=0).astype(np.float32)
        raise ValueError(f"Expected LiDAR keypoints with ndim 3 or 4, got shape {kpts.shape}.")

    @staticmethod
    def _transform_new_world_to_world(
        points: np.ndarray,
        pelvis: np.ndarray,
        r_new_to_world: np.ndarray,
        remove_root_rotation: bool = False,
    ) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        pel = np.asarray(pelvis, dtype=np.float32).reshape(1, 3)
        if not remove_root_rotation:
            return (pts + pel).reshape(points.shape).astype(np.float32)
        r = np.asarray(r_new_to_world, dtype=np.float32)
        if r.shape != (3, 3):
            raise ValueError(f"r_new_to_world must be (3,3), got {r.shape}.")
        out = (r @ pts.T).T + pel
        return out.reshape(points.shape).astype(np.float32)

    def _compute_new_world_transform(self, seq_name: str, start_frame: int) -> Tuple[np.ndarray, np.ndarray]:
        frame_window = self._build_frame_window(seq_name, start_frame)
        if self.causal:
            gt_body_frame_id = frame_window[-1]
        else:
            gt_body_frame_id = frame_window[len(frame_window) // 2]
        gt_keypoints_world = self._load_gt_keypoints(seq_name, gt_body_frame_id).astype(np.float32)
        pelvis = np.asarray(gt_keypoints_world[self.BODYCENTER_IDX], dtype=np.float32)
        r_new_to_world = np.eye(3, dtype=np.float32)
        if self.apply_to_new_world and self.remove_root_rotation:
            try:
                r_new_to_world = self._estimate_root_rotation_from_joints19(gt_keypoints_world)
            except ValueError:
                if self.root_rotation_fallback == "error":
                    raise
                r_new_to_world = np.eye(3, dtype=np.float32)
        return pelvis, r_new_to_world

    def _select_target_frame_from_sequence(self, kpts: np.ndarray) -> np.ndarray:
        arr = np.asarray(kpts, dtype=np.float32)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            t = arr.shape[0]
            idx = (self.seq_len - 1) if self.causal else (self.seq_len // 2)
            idx = int(np.clip(idx, 0, max(t - 1, 0)))
            return arr[idx]
        if arr.ndim == 4:
            v, t = int(arr.shape[0]), int(arr.shape[1])
            idx = (self.seq_len - 1) if self.causal else (self.seq_len // 2)
            idx = int(np.clip(idx, 0, max(t - 1, 0)))
            out = arr[:, idx]
            if v == 1:
                return out[0]
            return out
        raise ValueError(f"Unsupported LiDAR keypoint shape {arr.shape} for target-frame selection.")

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if not self.rgb_skeleton_index and not self.lidar_skeleton_index:
            return sample

        seq_name = sample.get("seq_name", None)
        start_frame = sample.get("start_frame", None)
        selected = sample.get("selected_cameras", {})
        rgb_cameras = selected.get("rgb", []) if isinstance(selected, dict) else []
        lidar_cameras = selected.get("lidar", []) if isinstance(selected, dict) else []
        if len(lidar_cameras) == 0 and isinstance(selected, dict):
            lidar_cameras = selected.get("depth", [])

        if seq_name is None or start_frame is None:
            parsed_seq, rgb_cam, depth_cam, body_frame_id = self._parse_sample_id(sample.get("sample_id", ""))
            if parsed_seq is None or body_frame_id is None:
                return sample
            seq_name = parsed_seq
            start_frame = self._body_frame_to_start_index(seq_name, body_frame_id)
            rgb_cameras = [rgb_cam] if rgb_cam is not None else []
            lidar_cameras = [depth_cam] if depth_cam is not None else []

        frame_window = self._build_frame_window(seq_name, int(start_frame))

        if self.rgb_skeleton_index:
            kpts_rgb = self._load_rgb_skeleton_frames_multi(seq_name, rgb_cameras, frame_window)
            image_size_hw = self._resolve_image_size_hw(sample)
            kpts_rgb = self._normalize_2d(kpts_rgb, image_size_hw)
            sample["gt_keypoints_2d_rgb"] = torch.from_numpy(kpts_rgb.astype(np.float32))

        if self.lidar_skeleton_index:
            kpts_lidar = self._load_lidar_skeleton_frames_multi(seq_name, lidar_cameras, frame_window)

            if not self.lidar_skeleton_is_pc_centered:
                if self.lidar_skeleton_coord == "lidar":
                    lidar_camera = sample.get("lidar_camera", None)
                    if lidar_camera is None:
                        lidar_camera = sample.get("depth_camera", None)
                    if lidar_camera is None:
                        raise ValueError(
                            "lidar_skeleton_coord='lidar' requires `lidar_camera` or `depth_camera` in sample."
                        )
                    kpts_lidar = self._transform_lidar_to_target_frame(kpts_lidar, lidar_camera)
                elif self.lidar_skeleton_coord == "new_world" and not self.apply_to_new_world:
                    pelvis, r_new_to_world = self._compute_new_world_transform(seq_name, int(start_frame))
                    kpts_lidar = self._transform_new_world_to_world(
                        kpts_lidar,
                        pelvis,
                        r_new_to_world,
                        remove_root_rotation=self.remove_root_rotation,
                    )
                elif self.lidar_skeleton_coord == "world" and self.apply_to_new_world:
                    pelvis, r_new_to_world = self._compute_new_world_transform(seq_name, int(start_frame))
                    if self.remove_root_rotation:
                        kpts_lidar = self._world_to_new_world_rot(kpts_lidar, pelvis, r_new_to_world)
                    else:
                        kpts_lidar = self._world_to_new_world(kpts_lidar, pelvis)

            if not self.return_keypoints_sequence:
                kpts_lidar = self._select_target_frame_from_sequence(kpts_lidar)
            sample[self.lidar_skeleton_key] = torch.from_numpy(kpts_lidar.astype(np.float32))

        return sample

