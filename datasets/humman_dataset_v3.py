import json
import os.path as osp
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from datasets.humman_dataset_v2 import HummanPreprocessedDatasetV2


class HummanPreprocessedDatasetV3(HummanPreprocessedDatasetV2):
    """Humman dataset v3.

    This class keeps v2 behavior and additionally supports loading skeleton
    predictions from JSON files as extra modalities.

    Expected JSON formats:
    1) {"predictions": [{"pc_path": "...", "keypoints": [[x,y,z], ...]}, ...]}
    2) {"relative/path.npy": [[x,y,z], ...], ...}

    Typical usage:
        modality_names=["rgb", "depth", "lidar_skl"]
        skeleton_json_modalities={"lidar_skl": "/path/to/lidar_skeletons.json"}
    """

    _BASE_MODALITIES = {"rgb", "depth", "lidar"}

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
        skeleton_json_modalities: Optional[Dict[str, str]] = None,
    ):
        requested_modalities = list(modality_names)
        base_modalities = [m for m in requested_modalities if m in self._BASE_MODALITIES]
        extra_modalities = [m for m in requested_modalities if m not in self._BASE_MODALITIES]

        if not base_modalities:
            raise ValueError(
                "HummanPreprocessedDatasetV3 requires at least one base modality "
                "from {'rgb','depth','lidar'} for sample indexing."
            )

        super().__init__(
            data_root=data_root,
            unit=unit,
            pipeline=pipeline,
            split=split,
            split_config=split_config,
            split_to_use=split_to_use,
            test_mode=test_mode,
            modality_names=base_modalities,
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

        self._sample_id_re = re.compile(
            r"^(?P<seq>p\d+_a\d+)_rgb_(?P<rgb>kinect_\d{3}|iphone|None)_depth_"
            r"(?P<depth>kinect_\d{3}|iphone|None)_(?P<start>\d+)$"
        )

        self.skeleton_json_modalities = skeleton_json_modalities or {}
        self._extra_modalities = []
        self._extra_index = {}
        self._extra_shape = {}

        for modality in extra_modalities:
            json_path = self.skeleton_json_modalities.get(modality)
            if json_path is None:
                warnings.warn(
                    f"Extra modality '{modality}' requested but no JSON path is provided. "
                    "This modality will be ignored."
                )
                continue

            index, shape = self._build_json_skeleton_index(json_path)
            if not index:
                warnings.warn(
                    f"No usable entries found for modality '{modality}' in {json_path}. "
                    "This modality will be ignored."
                )
                continue

            self._extra_modalities.append(modality)
            self._extra_index[modality] = index
            self._extra_shape[modality] = shape

        # Expose requested base + successfully loaded extra modalities.
        self.modality_names = base_modalities + self._extra_modalities

    def _build_json_skeleton_index(
        self, json_path: str
    ) -> Tuple[Dict[str, Dict[str, List[Tuple[int, np.ndarray]]]], Optional[Tuple[int, int]]]:
        path = Path(json_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict) and "predictions" in payload:
            entries = payload["predictions"]
        elif isinstance(payload, dict):
            entries = [{"pc_path": k, "keypoints": v} for k, v in payload.items()]
        elif isinstance(payload, list):
            entries = payload
        else:
            raise ValueError(f"Unsupported skeleton JSON format: {path}")

        index: Dict[str, Dict[str, List[Tuple[int, np.ndarray]]]] = {}
        inferred_shape: Optional[Tuple[int, int]] = None

        for item in entries:
            if not isinstance(item, dict):
                continue

            rel_path = item.get("pc_path") or item.get("path") or item.get("image_path")
            if rel_path is None:
                continue

            keypoints = item.get("keypoints")
            if keypoints is None and "instances" in item and item["instances"]:
                keypoints = item["instances"][0].get("keypoints")
            if keypoints is None:
                continue

            kpts = np.asarray(keypoints, dtype=np.float32)
            if kpts.ndim == 3 and kpts.shape[0] == 1:
                kpts = kpts[0]
            # Support both 2D and 3D keypoints from JSON predictions.
            if kpts.ndim != 2 or kpts.shape[1] < 2:
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

            index.setdefault(seq_name, {}).setdefault(camera, []).append((frame_idx, kpts))
            if inferred_shape is None:
                inferred_shape = (int(kpts.shape[0]), int(kpts.shape[1]))

        for seq_name in index:
            for camera in index[seq_name]:
                index[seq_name][camera].sort(key=lambda x: x[0])

        return index, inferred_shape

    def _choose_extra_camera(
        self,
        modality: str,
        seq_name: str,
        rgb_camera: Optional[str],
        depth_camera: Optional[str],
    ) -> Optional[str]:
        seq_data = self._extra_index[modality].get(seq_name, {})
        if not seq_data:
            return None

        if depth_camera and depth_camera in seq_data:
            return depth_camera
        if rgb_camera and rgb_camera in seq_data:
            return rgb_camera
        return sorted(seq_data.keys())[0]

    def _load_extra_skeleton_frames(
        self, modality: str, seq_name: str, camera: Optional[str], start_frame: int
    ) -> List[np.ndarray]:
        shape = self._extra_shape.get(modality, (24, 3))
        zeros = np.zeros(shape, dtype=np.float32)
        if camera is None:
            return [zeros.copy() for _ in range(self.seq_len)]

        frame_list = self._extra_index[modality].get(seq_name, {}).get(camera, [])
        if not frame_list:
            return [zeros.copy() for _ in range(self.seq_len)]

        frames: List[np.ndarray] = []
        for i in range(self.seq_len):
            idx = min(start_frame + i, len(frame_list) - 1)
            frames.append(frame_list[idx][1].copy())
        return frames

    def _parse_sample_id(self, sample_id: str):
        m = self._sample_id_re.match(sample_id)
        if m is None:
            return None, None, None, 0
        seq_name = m.group("seq")
        rgb_camera = None if m.group("rgb") == "None" else m.group("rgb")
        depth_camera = None if m.group("depth") == "None" else m.group("depth")
        start_frame = int(m.group("start"))
        return seq_name, rgb_camera, depth_camera, start_frame

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if not self._extra_modalities:
            return sample

        seq_name, rgb_camera, depth_camera, start_frame = self._parse_sample_id(
            sample.get("sample_id", "")
        )
        if seq_name is None:
            return sample

        for modality in self._extra_modalities:
            cam = self._choose_extra_camera(modality, seq_name, rgb_camera, depth_camera)
            sample[f"input_{modality}"] = self._load_extra_skeleton_frames(
                modality=modality,
                seq_name=seq_name,
                camera=cam,
                start_frame=start_frame,
            )
            if modality not in sample["modalities"]:
                sample["modalities"].append(modality)

        return sample
