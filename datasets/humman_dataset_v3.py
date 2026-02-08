import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from datasets.humman_dataset_v2 import HummanPreprocessedDatasetV2


class HummanPreprocessedDatasetV3(HummanPreprocessedDatasetV2):
    """V2 dataset with optional JSON-loaded 2D RGB skeletons.

    When `rgb_skeleton_json` is provided, this class loads 2D keypoints and
    writes them to `gt_keypoints_2d_rgb` (shape: [T, J, 2]), normalized to
    [-1, 1] in x/y.
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
        seq_len: int = 5,
        seq_step: int = 1,
        pad_seq: bool = False,
        causal: bool = False,
        use_all_pairs: bool = False,
        max_samples: Optional[int] = None,
        colocated: bool = False,
        convert_depth_to_lidar: bool = True,
        apply_to_new_world: bool = True,
        skeleton_only: bool = True,
        rgb_skeleton_json: Optional[str] = None,
        rgb_skeleton_image_size_hw: Sequence[int] = (512, 512),
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
            seq_len=seq_len,
            seq_step=seq_step,
            pad_seq=pad_seq,
            causal=causal,
            use_all_pairs=use_all_pairs,
            max_samples=max_samples,
            colocated=colocated,
            convert_depth_to_lidar=convert_depth_to_lidar,
            apply_to_new_world=apply_to_new_world,
            skeleton_only=skeleton_only,
        )

        self.rgb_skeleton_json = rgb_skeleton_json
        self.rgb_skeleton_image_size_hw = (
            int(rgb_skeleton_image_size_hw[0]),
            int(rgb_skeleton_image_size_hw[1]),
        )
        self.rgb_skeleton_index: Dict[str, Dict[str, List[Tuple[int, np.ndarray]]]] = {}
        self.rgb_skeleton_shape: Tuple[int, int] = (17, 2)

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

            stem = Path(rel_path).stem
            seq_match = self._seq_re.search(stem)
            cam_match = self._cam_re.search(stem)
            frame_match = self._frame_re.search(stem)
            if not (seq_match and cam_match and frame_match):
                continue

            seq_name = seq_match.group(1)
            camera = cam_match.group(1)
            frame_idx = int(frame_match.group(1))
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

    def _parse_sample_id(self, sample_id: str) -> Tuple[Optional[str], Optional[str], int]:
        m = self._sample_id_re.match(sample_id)
        if m is None:
            return None, None, 0
        seq_name = m.group("seq")
        rgb_camera = None if m.group("rgb") == "None" else m.group("rgb")
        start_frame = int(m.group("start"))
        return seq_name, rgb_camera, start_frame

    def _resolve_image_size_hw(self, sample) -> Tuple[int, int]:
        rgb = sample.get("input_rgb")
        if isinstance(rgb, torch.Tensor):
            if rgb.dim() >= 4:
                return int(rgb.shape[-2]), int(rgb.shape[-1])
        elif isinstance(rgb, np.ndarray):
            if rgb.ndim == 4:
                return int(rgb.shape[1]), int(rgb.shape[2])
            if rgb.ndim == 3:
                return int(rgb.shape[0]), int(rgb.shape[1])
        elif isinstance(rgb, (list, tuple)) and len(rgb) > 0:
            frame0 = rgb[0]
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

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if not self.rgb_skeleton_index:
            return sample

        seq_name, rgb_camera, start_frame = self._parse_sample_id(sample.get("sample_id", ""))
        if seq_name is None:
            return sample

        kpts = self._load_rgb_skeleton_frames(seq_name, rgb_camera, start_frame)
        image_size_hw = self._resolve_image_size_hw(sample)
        kpts = self._normalize_2d(kpts, image_size_hw)
        sample["gt_keypoints_2d_rgb"] = torch.from_numpy(kpts.astype(np.float32))
        return sample
