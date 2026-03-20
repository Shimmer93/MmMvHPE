from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from datasets.base_dataset import BaseDataset
from projects.synthetic_data.export_formats import (
    camera_to_pose_encoding,
    center_keypoints_with_pc,
    estimate_panoptic_root_rotation,
    transform_points_to_camera,
)


class SyntheticExportedTrainingDataset(BaseDataset):
    def __init__(
        self,
        data_root: str,
        target_format: str,
        pipeline: List[dict] = [],
        split: str = "train",
        test_mode: bool = False,
        modality_names: Sequence[str] = ("rgb", "lidar"),
        seq_len: int = 1,
        max_samples: Optional[int] = None,
        random_seed: int = 0,
    ):
        super().__init__(pipeline=pipeline)
        self.data_root = Path(data_root).expanduser().resolve()
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"Synthetic data_root not found: {self.data_root}")

        self.target_format = str(target_format).strip().lower()
        if self.target_format not in {"humman", "panoptic"}:
            raise ValueError(
                f"Unsupported target_format={target_format}. Expected one of ['humman', 'panoptic']."
            )

        self.split = str(split)
        self.test_mode = bool(test_mode)
        self.modality_names = [str(x).lower() for x in modality_names]
        invalid_modalities = [m for m in self.modality_names if m not in {"rgb", "lidar"}]
        if invalid_modalities:
            raise ValueError(
                f"SyntheticExportedTrainingDataset only supports modalities ['rgb', 'lidar'], "
                f"got invalid entries: {invalid_modalities}"
            )
        if int(seq_len) != 1:
            raise ValueError(
                f"SyntheticExportedTrainingDataset only supports seq_len=1, got seq_len={seq_len}."
            )
        self.seq_len = 1
        self.max_samples = max_samples
        self.random_seed = int(random_seed)

        self.data_list = self._index_samples()
        if self.max_samples is not None:
            if int(self.max_samples) <= 0:
                self.data_list = []
            elif len(self.data_list) > int(self.max_samples):
                rng = np.random.RandomState(self.random_seed)
                idxs = rng.choice(len(self.data_list), size=int(self.max_samples), replace=False)
                self.data_list = [self.data_list[int(i)] for i in sorted(idxs.tolist())]

    def _index_samples(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for sample_dir in sorted(p for p in self.data_root.iterdir() if p.is_dir() and p.name.startswith("ann_")):
            source_manifest_path = sample_dir / "manifest.json"
            export_manifest_path = sample_dir / "exports" / "export_manifest.json"
            if not source_manifest_path.is_file() or not export_manifest_path.is_file():
                continue

            source_manifest = self._load_json(source_manifest_path)
            export_manifest = self._load_json(export_manifest_path)
            if source_manifest.get("status") != "accepted":
                continue
            if export_manifest.get("status") != "accepted":
                continue

            format_manifest_path = sample_dir / "exports" / self.target_format / "manifest.json"
            if not format_manifest_path.is_file():
                continue
            format_manifest = self._load_json(format_manifest_path)
            if format_manifest.get("status") != "accepted":
                continue

            samples.append(
                {
                    "sample_dir": sample_dir,
                    "source_manifest_path": source_manifest_path,
                    "source_manifest": source_manifest,
                    "export_manifest_path": export_manifest_path,
                    "export_manifest": export_manifest,
                    "format_manifest_path": format_manifest_path,
                    "format_manifest": format_manifest,
                }
            )
        return samples

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to load JSON from {path}: {exc}") from exc

    @staticmethod
    def _load_npy(path: Path) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(f"Required artifact not found: {path}")
        arr = np.load(path)
        return np.asarray(arr)

    @staticmethod
    def _load_camera_json(path: Path) -> dict[str, np.ndarray]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "intrinsic" not in payload or "extrinsic" not in payload:
            raise ValueError(f"Camera JSON missing intrinsic/extrinsic: {path}")
        return {
            "intrinsic": np.asarray(payload["intrinsic"], dtype=np.float32).reshape(3, 3),
            "extrinsic": np.asarray(payload["extrinsic"], dtype=np.float32).reshape(3, 4),
        }

    @staticmethod
    def _load_rgb_image(path: Path) -> np.ndarray:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"RGB image not found: {path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _resolve_artifact_path(value: str) -> Path:
        return Path(value).expanduser().resolve()

    def __len__(self) -> int:
        return len(self.data_list)

    def _load_common_sample(self, entry: dict[str, Any]) -> dict[str, Any]:
        sample_dir = Path(entry["sample_dir"]).resolve()
        source_manifest = entry["source_manifest"]
        image_path = Path(source_manifest["image_path"]).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Source RGB image missing for synthetic sample {sample_dir.name}: {image_path}"
            )

        sample: dict[str, Any] = {
            "sample_id": sample_dir.name,
            "seq_name": sample_dir.name,
            "start_frame": 0,
            "modalities": list(self.modality_names),
            "selected_cameras": {
                "rgb": ["synthetic_rgb"] if "rgb" in self.modality_names else [],
                "depth": [],
                "lidar": ["synthetic_lidar"] if "lidar" in self.modality_names else [],
            },
            "source_dataset": source_manifest.get("source_dataset"),
            "source_image_path": str(image_path),
            "source_annotation_id": int(source_manifest.get("annotation_id", -1)),
            "source_image_id": int(source_manifest.get("image_id", -1)),
        }

        if "rgb" in self.modality_names:
            sample["input_rgb"] = [self._load_rgb_image(image_path)]

        if "lidar" in self.modality_names:
            lidar_path = self._resolve_artifact_path(
                source_manifest["artifacts"]["synthetic_lidar_points_sensor"]
            )
            sample["input_lidar"] = [self._load_npy(lidar_path).astype(np.float32)]

        return sample

    def _load_humman_sample(self, entry: dict[str, Any]) -> dict[str, Any]:
        sample = self._load_common_sample(entry)
        export_dir = Path(entry["sample_dir"]).resolve() / "exports" / "humman"
        format_manifest = entry["format_manifest"]
        artifacts = format_manifest.get("artifacts", {})

        required = [
            "gt_keypoints",
            "gt_smpl_params",
            "gt_global_orient",
            "gt_pelvis",
            "gt_keypoints_lidar",
            "gt_keypoints_pc_centered_input_lidar",
            "rgb_camera",
            "lidar_camera",
            "gt_keypoints_2d_rgb",
        ]
        missing = [key for key in required if key not in artifacts]
        if missing:
            raise ValueError(
                f"HuMMan synthetic export for {entry['sample_dir'].name} is missing artifacts: {missing}"
            )

        sample["gt_keypoints"] = self._load_npy(self._resolve_artifact_path(artifacts["gt_keypoints"])).astype(
            np.float32
        )
        sample["gt_smpl_params"] = self._load_npy(
            self._resolve_artifact_path(artifacts["gt_smpl_params"])
        ).astype(np.float32)
        sample["gt_global_orient"] = self._load_npy(
            self._resolve_artifact_path(artifacts["gt_global_orient"])
        ).astype(np.float32)
        sample["gt_pelvis"] = self._load_npy(self._resolve_artifact_path(artifacts["gt_pelvis"])).astype(np.float32)
        sample["gt_keypoints_lidar"] = self._load_npy(
            self._resolve_artifact_path(artifacts["gt_keypoints_lidar"])
        ).astype(np.float32)
        sample["gt_keypoints_pc_centered_input_lidar"] = self._load_npy(
            self._resolve_artifact_path(artifacts["gt_keypoints_pc_centered_input_lidar"])
        ).astype(np.float32)
        sample["gt_keypoints_2d_rgb"] = self._load_npy(
            self._resolve_artifact_path(artifacts["gt_keypoints_2d_rgb"])
        ).astype(np.float32)
        sample["rgb_camera"] = self._load_camera_json(self._resolve_artifact_path(artifacts["rgb_camera"]))
        sample["lidar_camera"] = self._load_camera_json(self._resolve_artifact_path(artifacts["lidar_camera"]))
        sample["export_format"] = "humman"
        sample["export_dir"] = str(export_dir)
        return sample

    def _derive_panoptic_lidar_payload(self, entry: dict[str, Any], sample: dict[str, Any]) -> None:
        source_manifest = entry["source_manifest"]
        artifacts = source_manifest.get("artifacts", {})
        panoptic_world = np.asarray(sample["gt_keypoints_world"], dtype=np.float32)
        panoptic_new_world = np.asarray(sample["gt_keypoints"], dtype=np.float32)
        body_center = np.asarray(sample["gt_pelvis"], dtype=np.float32).reshape(3)

        if "lidar_extrinsic_world_to_sensor" not in artifacts:
            raise ValueError(
                f"Base synthetic manifest for {entry['sample_dir'].name} is missing lidar_extrinsic_world_to_sensor."
            )
        if "pelvis_source_frame" not in artifacts:
            raise ValueError(
                f"Base synthetic manifest for {entry['sample_dir'].name} is missing pelvis_source_frame."
            )

        lidar_extrinsic_canonical = self._load_npy(
            self._resolve_artifact_path(artifacts["lidar_extrinsic_world_to_sensor"])
        ).astype(np.float32)
        if lidar_extrinsic_canonical.shape != (3, 4):
            raise ValueError(
                f"Expected lidar_extrinsic_world_to_sensor shape (3,4), got {lidar_extrinsic_canonical.shape} "
                f"for {entry['sample_dir'].name}."
            )

        mhr_pelvis = self._load_npy(self._resolve_artifact_path(artifacts["pelvis_source_frame"])).astype(
            np.float32
        )
        if mhr_pelvis.shape != (3,):
            mhr_pelvis = np.asarray(mhr_pelvis, dtype=np.float32).reshape(3)

        r_new_to_world = estimate_panoptic_root_rotation(panoptic_world)
        r_wc = lidar_extrinsic_canonical[:, :3]
        t_wc = lidar_extrinsic_canonical[:, 3:4]
        offset_world = (body_center - mhr_pelvis).reshape(3, 1)
        lidar_extrinsic_panoptic_new = np.hstack(
            [r_wc @ r_new_to_world, r_wc @ offset_world + t_wc]
        ).astype(np.float32)
        lidar_camera = {
            "intrinsic": np.eye(3, dtype=np.float32),
            "extrinsic": lidar_extrinsic_panoptic_new,
        }

        if "input_lidar" not in sample:
            raise ValueError(
                f"Panoptic synthetic sample {entry['sample_dir'].name} requires input_lidar to derive LiDAR targets."
            )
        input_lidar = np.asarray(sample["input_lidar"][0], dtype=np.float32)
        gt_keypoints_lidar = transform_points_to_camera(panoptic_new_world, lidar_extrinsic_panoptic_new)
        gt_keypoints_pc_centered, _ = center_keypoints_with_pc(gt_keypoints_lidar, input_lidar)

        sample["lidar_camera"] = lidar_camera
        sample["gt_keypoints_lidar"] = gt_keypoints_lidar.astype(np.float32)
        sample["gt_keypoints_pc_centered_input_lidar"] = gt_keypoints_pc_centered.astype(np.float32)
        sample["gt_camera_lidar"] = camera_to_pose_encoding(
            lidar_camera, image_size_hw=(1, int(input_lidar.shape[0]))
        ).astype(np.float32)

    def _load_panoptic_sample(self, entry: dict[str, Any]) -> dict[str, Any]:
        sample = self._load_common_sample(entry)
        format_manifest = entry["format_manifest"]
        artifacts = format_manifest.get("artifacts", {})

        required = [
            "gt_keypoints",
            "gt_keypoints_world",
            "gt_pelvis",
            "rgb_camera",
            "gt_keypoints_2d_rgb",
        ]
        missing = [key for key in required if key not in artifacts]
        if missing:
            raise ValueError(
                f"Panoptic synthetic export for {entry['sample_dir'].name} is missing artifacts: {missing}"
            )

        sample["gt_keypoints"] = self._load_npy(self._resolve_artifact_path(artifacts["gt_keypoints"])).astype(
            np.float32
        )
        sample["gt_keypoints_world"] = self._load_npy(
            self._resolve_artifact_path(artifacts["gt_keypoints_world"])
        ).astype(np.float32)
        sample["gt_pelvis"] = self._load_npy(self._resolve_artifact_path(artifacts["gt_pelvis"])).astype(np.float32)
        sample["gt_global_orient"] = np.zeros((3,), dtype=np.float32)
        sample["gt_smpl_params"] = np.zeros((82,), dtype=np.float32)
        sample["gt_keypoints_2d_rgb"] = self._load_npy(
            self._resolve_artifact_path(artifacts["gt_keypoints_2d_rgb"])
        ).astype(np.float32)
        sample["rgb_camera"] = self._load_camera_json(self._resolve_artifact_path(artifacts["rgb_camera"]))
        sample["export_format"] = "panoptic"
        self._derive_panoptic_lidar_payload(entry, sample)
        return sample

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.data_list[int(index)]
        if self.target_format == "humman":
            sample = self._load_humman_sample(entry)
        elif self.target_format == "panoptic":
            sample = self._load_panoptic_sample(entry)
        else:
            raise AssertionError(f"Unexpected target_format={self.target_format}")
        sample = self.pipeline(sample)
        return sample
