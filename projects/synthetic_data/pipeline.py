from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .coco_adapter import COCOValPersonAdapter
from .sam3d_adapter import SAM3DRunner, validate_sam3d_paths
from .virtual_lidar import (
    VirtualLidarConfig,
    sample_virtual_lidar_pose,
    sample_visible_surface_pointcloud,
)
from .visualization import save_mask_image, save_rgb_image, save_summary_figure

MHR70_LEFT_HIP_IDX = 9
MHR70_RIGHT_HIP_IDX = 10


@dataclass(frozen=True)
class SyntheticGenerationConfig:
    data_root: str = "/opt/data/coco"
    annotation_file: str = "annotations/person_keypoints_val2017.json"
    image_dir: str = "val2017"
    checkpoint_root: str = "/opt/data/SAM_3dbody_checkpoints"
    output_root: str = "logs/synthetic_data/v0a"
    min_area: float = 4096.0
    min_keypoints: int = 5
    one_person_only: bool = True
    min_mask_pixels: int = 1024
    seed: int = 42
    device: str = "cuda"
    save_source_rgb: bool = False
    save_visualizations: bool = False
    lidar: VirtualLidarConfig = VirtualLidarConfig()


class SyntheticGenerationPipeline:
    def __init__(self, repo_root: str | Path, cfg: SyntheticGenerationConfig) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.cfg = cfg
        self.output_root = Path(cfg.output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)

        validate_sam3d_paths(cfg.checkpoint_root)
        self.adapter = COCOValPersonAdapter(
            cfg.data_root,
            annotation_file=cfg.annotation_file,
            image_dir=cfg.image_dir,
            min_area=cfg.min_area,
            min_keypoints=cfg.min_keypoints,
            one_person_only=cfg.one_person_only,
        )
        self.sam3d = SAM3DRunner(
            repo_root=self.repo_root,
            checkpoint_root=cfg.checkpoint_root,
            device=cfg.device,
        )

    def _sample_dir(self, annotation_id: int, image_id: int) -> Path:
        return self.output_root / f"ann_{annotation_id:012d}_img_{image_id:012d}"

    def __len__(self) -> int:
        return len(self.adapter)

    def sample_dir_for_index(self, index: int) -> Path:
        record = self.adapter.get_record(index)
        return self._sample_dir(record.annotation_id, record.image_id)

    def manifest_path_for_index(self, index: int) -> Path:
        return self.sample_dir_for_index(index) / "manifest.json"

    @staticmethod
    def _canonicalize_mhr70(keypoints_3d: np.ndarray, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        keypoints_3d = np.asarray(keypoints_3d, dtype=np.float32)
        vertices = np.asarray(vertices, dtype=np.float32)
        if keypoints_3d.ndim != 2 or keypoints_3d.shape[1] != 3:
            raise ValueError(f"Expected keypoints shape (J,3), got {keypoints_3d.shape}")
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"Expected vertices shape (V,3), got {vertices.shape}")
        if keypoints_3d.shape[0] <= max(MHR70_LEFT_HIP_IDX, MHR70_RIGHT_HIP_IDX):
            raise ValueError(f"MHR70 keypoints missing hip indices: shape={keypoints_3d.shape}")
        pelvis = 0.5 * (
            keypoints_3d[MHR70_LEFT_HIP_IDX] + keypoints_3d[MHR70_RIGHT_HIP_IDX]
        )
        canonical_keypoints = keypoints_3d - pelvis[None, :]
        canonical_vertices = vertices - pelvis[None, :]
        return canonical_keypoints.astype(np.float32), canonical_vertices.astype(np.float32), pelvis.astype(np.float32)

    def _quality_check(self, *, mask: np.ndarray, sam3d_output: dict[str, Any]) -> None:
        mask_pixels = int(np.count_nonzero(mask))
        if mask_pixels < self.cfg.min_mask_pixels:
            raise ValueError(
                f"Mask area too small: {mask_pixels} pixels < min_mask_pixels={self.cfg.min_mask_pixels}."
            )
        for key in ["pred_keypoints_3d", "pred_vertices", "pred_cam_t", "focal_length"]:
            if key not in sam3d_output:
                raise ValueError(f"SAM-3D-Body output is missing required key `{key}`.")
            arr = np.asarray(sam3d_output[key])
            if not np.isfinite(arr).all():
                raise ValueError(f"SAM-3D-Body output `{key}` contains non-finite values.")
        if np.asarray(sam3d_output["pred_vertices"]).shape[0] < 1000:
            raise ValueError("Reconstructed mesh has too few vertices.")

    def _save_arrays(
        self,
        sample_dir: Path,
        *,
        sam3d_output: dict[str, Any],
        canonical_keypoints: np.ndarray,
        canonical_vertices: np.ndarray,
        pelvis_source_frame: np.ndarray,
        pointcloud_sensor: np.ndarray,
        lidar_pose: dict[str, Any],
    ) -> dict[str, str]:
        array_dir = sample_dir / "arrays"
        array_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "pred_keypoints_3d_raw": array_dir / "pred_keypoints_3d_raw.npy",
            "pred_vertices_raw": array_dir / "pred_vertices_raw.npy",
            "pred_keypoints_3d_canonical": array_dir / "pred_keypoints_3d_canonical.npy",
            "pred_vertices_canonical": array_dir / "pred_vertices_canonical.npy",
            "pelvis_source_frame": array_dir / "pelvis_source_frame.npy",
            "synthetic_lidar_points_sensor": array_dir / "synthetic_lidar_points_sensor.npy",
            "mesh_faces": array_dir / "mesh_faces.npy",
            "lidar_extrinsic_world_to_sensor": array_dir / "lidar_extrinsic_world_to_sensor.npy",
        }
        np.save(paths["pred_keypoints_3d_raw"], np.asarray(sam3d_output["pred_keypoints_3d"], dtype=np.float32))
        np.save(paths["pred_vertices_raw"], np.asarray(sam3d_output["pred_vertices"], dtype=np.float32))
        np.save(paths["pred_keypoints_3d_canonical"], canonical_keypoints)
        np.save(paths["pred_vertices_canonical"], canonical_vertices)
        np.save(paths["pelvis_source_frame"], pelvis_source_frame)
        np.save(paths["synthetic_lidar_points_sensor"], pointcloud_sensor)
        np.save(paths["mesh_faces"], np.asarray(sam3d_output["faces"], dtype=np.int32))
        np.save(
            paths["lidar_extrinsic_world_to_sensor"],
            np.asarray(lidar_pose["extrinsic_world_to_sensor"], dtype=np.float32),
        )
        return {key: str(value) for key, value in paths.items()}

    def process_index(self, index: int) -> dict[str, Any]:
        source = self.adapter.load_sample(index)
        record = source["record"]
        sample_dir = self._sample_dir(record.annotation_id, record.image_id)
        sample_dir.mkdir(parents=True, exist_ok=True)

        manifest: dict[str, Any] = {
            "status": "rejected",
            "rejection_reason": None,
            "source_dataset": "coco_val2017",
            "source_data_root": str(Path(self.cfg.data_root).expanduser().resolve()),
            "annotation_index": int(index),
            "annotation_id": int(record.annotation_id),
            "image_id": int(record.image_id),
            "image_path": str(record.image_path),
            "image_size_hw": [int(record.image_height), int(record.image_width)],
            "bbox_xywh": record.bbox_xywh.astype(float).tolist(),
            "mask_provenance": source["mask_provenance"],
            "checkpoint_root": str(Path(self.cfg.checkpoint_root).expanduser().resolve()),
            "mask_assisted_sam3d": True,
            "simulation_mode": "visible_surface_sampling_v0a",
            "save_source_rgb": bool(self.cfg.save_source_rgb),
            "save_visualizations": bool(self.cfg.save_visualizations),
            "artifacts": {},
        }
        manifest_path = sample_dir / "manifest.json"

        try:
            image_rgb = source["image_rgb"]
            bbox_xyxy = np.asarray(source["bbox_xyxy"], dtype=np.float32)
            mask = np.asarray(source["mask"], dtype=np.uint8)

            save_mask_image(sample_dir / "source_mask.png", mask)
            manifest["artifacts"]["source_mask"] = str(sample_dir / "source_mask.png")
            if self.cfg.save_source_rgb:
                save_rgb_image(sample_dir / "source_rgb.png", image_rgb)
                manifest["artifacts"]["source_rgb"] = str(sample_dir / "source_rgb.png")

            sam3d_output = self.sam3d.infer(image_rgb=image_rgb, bbox_xyxy=bbox_xyxy, mask=mask)
            self._quality_check(mask=mask, sam3d_output=sam3d_output)

            canonical_keypoints, canonical_vertices, pelvis = self._canonicalize_mhr70(
                keypoints_3d=np.asarray(sam3d_output["pred_keypoints_3d"], dtype=np.float32),
                vertices=np.asarray(sam3d_output["pred_vertices"], dtype=np.float32),
            )

            rng = random.Random(self.cfg.seed + int(record.annotation_id))
            lidar_pose = sample_virtual_lidar_pose(rng=rng, cfg=self.cfg.lidar)
            pointcloud_sensor = sample_visible_surface_pointcloud(
                vertices_world=canonical_vertices,
                faces=np.asarray(sam3d_output["faces"], dtype=np.int32),
                sensor_position_world=np.asarray(lidar_pose["sensor_position_world"], dtype=np.float32),
                extrinsic_world_to_sensor=np.asarray(lidar_pose["extrinsic_world_to_sensor"], dtype=np.float32),
                cfg=self.cfg.lidar,
                rng=rng,
            )

            array_paths = self._save_arrays(
                sample_dir,
                sam3d_output=sam3d_output,
                canonical_keypoints=canonical_keypoints,
                canonical_vertices=canonical_vertices,
                pelvis_source_frame=pelvis,
                pointcloud_sensor=pointcloud_sensor,
                lidar_pose=lidar_pose,
            )
            manifest["artifacts"].update(array_paths)

            if self.cfg.save_visualizations:
                overlay_rgb = self.sam3d.render_overlay(image_rgb=image_rgb, output=sam3d_output)
                save_rgb_image(sample_dir / "sam3d_overlay.png", overlay_rgb)
                manifest["artifacts"]["sam3d_overlay"] = str(sample_dir / "sam3d_overlay.png")

                summary_path = sample_dir / "summary.png"
                save_summary_figure(
                    summary_path,
                    image_rgb=image_rgb,
                    mask=mask,
                    reconstruction_overlay_rgb=overlay_rgb,
                    canonical_keypoints=canonical_keypoints,
                    canonical_vertices=canonical_vertices,
                    sensor_position_world=np.asarray(lidar_pose["sensor_position_world"], dtype=np.float32),
                    pointcloud_sensor=pointcloud_sensor,
                )
                manifest["artifacts"]["summary"] = str(summary_path)

            manifest["status"] = "accepted"
            manifest["canonical_frame"] = {
                "name": "pelvis_centered_mhr70",
                "pelvis_indices": [MHR70_LEFT_HIP_IDX, MHR70_RIGHT_HIP_IDX],
                "pelvis_source_frame": pelvis.astype(float).tolist(),
            }
            manifest["lidar_pose"] = {
                "sensor_position_world": np.asarray(lidar_pose["sensor_position_world"], dtype=np.float32).astype(float).tolist(),
                "target_world": np.asarray(lidar_pose["target_world"], dtype=np.float32).astype(float).tolist(),
                "radius": float(lidar_pose["radius"]),
                "azimuth_deg": float(lidar_pose["azimuth_deg"]),
                "elevation_deg": float(lidar_pose["elevation_deg"]),
            }
            manifest["lidar_sampling"] = {
                "num_points": int(self.cfg.lidar.num_points),
                "oversample_factor": int(self.cfg.lidar.oversample_factor),
                "surface_noise_std": float(self.cfg.lidar.surface_noise_std),
            }
        except Exception as exc:
            manifest["rejection_reason"] = str(exc)

        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["manifest_path"] = str(manifest_path)
        return manifest

    def process_range(self, start_index: int, max_samples: int) -> list[dict[str, Any]]:
        if max_samples <= 0:
            raise ValueError(f"max_samples must be > 0, got {max_samples}")
        results = []
        for index in range(start_index, min(start_index + max_samples, len(self.adapter))):
            results.append(self.process_index(index))
        return results
