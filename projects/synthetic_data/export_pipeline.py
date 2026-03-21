from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .export_formats import (
    TOPOLOGY_METADATA,
    axis_angle_to_matrix_np,
    build_rgb_camera_from_sam3d,
    camera_to_pose_encoding,
    center_keypoints_with_pc,
    estimate_panoptic_root_rotation,
    ensure_relative_symlink,
    mhr70_to_panoptic19,
    normalize_points_2d,
    project_points_to_image,
    save_camera_json,
    save_numpy,
    transform_points_to_camera,
    update_extrinsic_for_new_world,
    world_to_new_world,
)
from .mhr_smpl_adapter import MHRSMPLAdapter, MHRSMPLFitConfig
from .sam3d_adapter import SAM3DRunner, validate_sam3d_paths


@dataclass(frozen=True)
class SyntheticTargetExportConfig:
    synthetic_root: str
    checkpoint_root: str = "/opt/data/SAM_3dbody_checkpoints"
    export_dirname: str = "exports"
    save_pose_encodings: bool = True
    save_rgb_2d_keypoints: bool = True
    device: str = "cuda"
    lidar_version: str = "v0a"
    mhr_smpl: MHRSMPLFitConfig = MHRSMPLFitConfig()


class SyntheticTargetExportPipeline:
    def __init__(self, repo_root: str | Path, cfg: SyntheticTargetExportConfig) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.cfg = cfg
        self.synthetic_root = Path(cfg.synthetic_root).expanduser().resolve()
        if not self.synthetic_root.is_dir():
            raise FileNotFoundError(f"Synthetic root not found: {self.synthetic_root}")
        validate_sam3d_paths(cfg.checkpoint_root)
        self.sam3d = SAM3DRunner(
            repo_root=self.repo_root,
            checkpoint_root=cfg.checkpoint_root,
            device=cfg.device,
        )
        self.smpl_adapter = MHRSMPLAdapter(repo_root=self.repo_root, cfg=cfg.mhr_smpl)

    def list_sample_dirs(self) -> list[Path]:
        return sorted(
            p for p in self.synthetic_root.iterdir() if p.is_dir() and p.name.startswith("ann_")
        )

    @staticmethod
    def _load_manifest(sample_dir: Path) -> dict[str, Any]:
        manifest_path = sample_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Sample manifest not found: {manifest_path}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    @staticmethod
    def _load_mask(sample_dir: Path) -> np.ndarray:
        mask_path = sample_dir / "source_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Sample mask not found: {mask_path}")
        return mask.astype(np.uint8)

    @staticmethod
    def _load_image(image_path: str) -> np.ndarray:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _bbox_xywh_to_xyxy(bbox_xywh: list[float] | np.ndarray) -> np.ndarray:
        bbox = np.asarray(bbox_xywh, dtype=np.float32).reshape(4)
        return np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32)

    def _resolve_lidar_artifact_path(self, base_manifest: dict[str, Any]) -> Path:
        lidar_version = str(self.cfg.lidar_version).strip().lower()
        if not lidar_version:
            raise ValueError("SyntheticTargetExportConfig.lidar_version must be a non-empty string.")
        lidar_artifacts = base_manifest.get("lidar_artifacts", {})
        if lidar_artifacts:
            if lidar_version not in lidar_artifacts:
                raise ValueError(
                    f"Requested lidar_version={lidar_version}, but available versions are "
                    f"{sorted(lidar_artifacts.keys())}."
                )
            artifact_path = lidar_artifacts[lidar_version].get("artifact_path", None)
            if not artifact_path:
                raise ValueError(
                    f"Manifest lidar_artifacts[{lidar_version}] is missing artifact_path."
                )
            return Path(artifact_path).expanduser().resolve()
        if lidar_version != "v0a":
            raise ValueError(
                f"Requested lidar_version={lidar_version}, but manifest only supports legacy v0a layout."
            )
        return Path(base_manifest["artifacts"]["synthetic_lidar_points_sensor"]).expanduser().resolve()

    @staticmethod
    def _save_manifest(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _rerun_sam3d(self, sample_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
        image_rgb = self._load_image(manifest["image_path"])
        mask = self._load_mask(sample_dir)
        bbox_xyxy = self._bbox_xywh_to_xyxy(manifest["bbox_xywh"])
        return self.sam3d.infer(image_rgb=image_rgb, bbox_xyxy=bbox_xyxy, mask=mask)

    def _build_humman_export(
        self,
        sample_dir: Path,
        base_manifest: dict[str, Any],
        sam3d_output: dict[str, Any],
    ) -> dict[str, Any]:
        image_size_hw = tuple(int(x) for x in base_manifest["image_size_hw"])
        mhr_vertices_world = (
            np.asarray(sam3d_output["pred_vertices"], dtype=np.float32)
            + np.asarray(sam3d_output["pred_cam_t"], dtype=np.float32)[None, :]
        )
        mhr_keypoints_world = (
            np.asarray(sam3d_output["pred_keypoints_3d"], dtype=np.float32)
            + np.asarray(sam3d_output["pred_cam_t"], dtype=np.float32)[None, :]
        )
        smpl_fit = self.smpl_adapter.fit_smpl_to_sam3d_output(sam3d_output)

        smpl_joints_world = np.asarray(smpl_fit["smpl_joints24_world"], dtype=np.float32)
        smpl_vertices_world = np.asarray(smpl_fit["smpl_vertices_world"], dtype=np.float32)
        global_orient = np.asarray(smpl_fit["global_orient"], dtype=np.float32)
        pelvis_world = np.asarray(smpl_joints_world[0], dtype=np.float32)
        R_new_to_world = axis_angle_to_matrix_np(global_orient)
        smpl_joints_new_world = world_to_new_world(smpl_joints_world, pelvis_world, R_new_to_world)
        smpl_vertices_new_world = world_to_new_world(smpl_vertices_world, pelvis_world, R_new_to_world)

        rgb_camera_world = build_rgb_camera_from_sam3d(
            image_size_hw=image_size_hw,
            focal_length=sam3d_output["focal_length"],
            extrinsic_world_to_camera=np.hstack([np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)]),
        )
        rgb_camera_new_world = {
            "intrinsic": rgb_camera_world["intrinsic"],
            "extrinsic": update_extrinsic_for_new_world(
                rgb_camera_world["extrinsic"], pelvis_world, R_new_to_world
            ),
        }

        lidar_extrinsic_world_to_sensor_canonical = np.load(
            base_manifest["artifacts"]["lidar_extrinsic_world_to_sensor"]
        ).astype(np.float32)
        lidar_extrinsic_new_world_to_sensor = np.hstack(
            [
                lidar_extrinsic_world_to_sensor_canonical[:, :3] @ R_new_to_world,
                lidar_extrinsic_world_to_sensor_canonical[:, 3:4],
            ]
        ).astype(np.float32)
        lidar_camera_new_world = {
            "intrinsic": np.eye(3, dtype=np.float32),
            "extrinsic": lidar_extrinsic_new_world_to_sensor,
        }

        input_lidar_path = self._resolve_lidar_artifact_path(base_manifest)
        input_lidar_sensor = np.load(input_lidar_path).astype(np.float32)
        gt_keypoints_lidar = transform_points_to_camera(smpl_joints_new_world, lidar_camera_new_world["extrinsic"])
        gt_keypoints_pc_centered_input_lidar, gt_keypoints_pc_center_lidar = center_keypoints_with_pc(
            gt_keypoints_lidar, input_lidar_sensor
        )

        export_dir = sample_dir / self.cfg.export_dirname / "humman"
        export_dir.mkdir(parents=True, exist_ok=True)
        ensure_relative_symlink(export_dir / "input_lidar.npy", input_lidar_path)
        save_numpy(export_dir / "gt_keypoints.npy", smpl_joints_new_world)
        save_numpy(export_dir / "gt_keypoints_world.npy", smpl_joints_world)
        save_numpy(export_dir / "mhr_keypoints_world.npy", mhr_keypoints_world)
        save_numpy(export_dir / "mhr_vertices_world.npy", mhr_vertices_world)
        save_numpy(export_dir / "gt_vertices_world.npy", smpl_vertices_world)
        save_numpy(export_dir / "gt_vertices_new_world.npy", smpl_vertices_new_world)
        save_numpy(export_dir / "gt_smpl_params.npy", np.asarray(smpl_fit["gt_smpl_params"], dtype=np.float32))
        save_numpy(export_dir / "gt_global_orient.npy", global_orient)
        save_numpy(export_dir / "gt_pelvis.npy", pelvis_world)
        save_numpy(export_dir / "gt_keypoints_lidar.npy", gt_keypoints_lidar)
        save_numpy(export_dir / "gt_keypoints_pc_centered_input_lidar.npy", gt_keypoints_pc_centered_input_lidar)
        save_numpy(export_dir / "gt_keypoints_pc_center_lidar.npy", gt_keypoints_pc_center_lidar)
        save_numpy(export_dir / "smpl_faces.npy", np.asarray(smpl_fit["smpl_faces"], dtype=np.int32))
        save_camera_json(export_dir / "rgb_camera.json", rgb_camera_new_world)
        save_camera_json(export_dir / "lidar_camera.json", lidar_camera_new_world)

        artifacts: dict[str, str] = {
            "input_lidar": str(export_dir / "input_lidar.npy"),
            "gt_keypoints": str(export_dir / "gt_keypoints.npy"),
            "gt_keypoints_world": str(export_dir / "gt_keypoints_world.npy"),
            "mhr_keypoints_world": str(export_dir / "mhr_keypoints_world.npy"),
            "mhr_vertices_world": str(export_dir / "mhr_vertices_world.npy"),
            "gt_vertices_world": str(export_dir / "gt_vertices_world.npy"),
            "gt_vertices_new_world": str(export_dir / "gt_vertices_new_world.npy"),
            "gt_smpl_params": str(export_dir / "gt_smpl_params.npy"),
            "gt_global_orient": str(export_dir / "gt_global_orient.npy"),
            "gt_pelvis": str(export_dir / "gt_pelvis.npy"),
            "gt_keypoints_lidar": str(export_dir / "gt_keypoints_lidar.npy"),
            "gt_keypoints_pc_centered_input_lidar": str(export_dir / "gt_keypoints_pc_centered_input_lidar.npy"),
            "gt_keypoints_pc_center_lidar": str(export_dir / "gt_keypoints_pc_center_lidar.npy"),
            "smpl_faces": str(export_dir / "smpl_faces.npy"),
            "rgb_camera": str(export_dir / "rgb_camera.json"),
            "lidar_camera": str(export_dir / "lidar_camera.json"),
        }

        if self.cfg.save_pose_encodings:
            gt_camera_rgb = camera_to_pose_encoding(rgb_camera_new_world, image_size_hw)
            lidar_pose_hw = (1, int(input_lidar_sensor.shape[0]))
            gt_camera_lidar = camera_to_pose_encoding(lidar_camera_new_world, lidar_pose_hw)
            save_numpy(export_dir / "gt_camera_rgb.npy", gt_camera_rgb)
            save_numpy(export_dir / "gt_camera_lidar.npy", gt_camera_lidar)
            artifacts["gt_camera_rgb"] = str(export_dir / "gt_camera_rgb.npy")
            artifacts["gt_camera_lidar"] = str(export_dir / "gt_camera_lidar.npy")

        if self.cfg.save_rgb_2d_keypoints:
            gt_keypoints_2d_rgb = normalize_points_2d(
                project_points_to_image(smpl_joints_world, rgb_camera_world),
                image_size_hw=image_size_hw,
            )
            save_numpy(export_dir / "gt_keypoints_2d_rgb.npy", gt_keypoints_2d_rgb.astype(np.float32))
            artifacts["gt_keypoints_2d_rgb"] = str(export_dir / "gt_keypoints_2d_rgb.npy")

        metadata = {
            "format_name": "humman_smpl24",
            "status": "accepted",
            "topology": {
                "gt_keypoints": TOPOLOGY_METADATA["smpl24_new_world"].__dict__,
                "gt_keypoints_world": TOPOLOGY_METADATA["smpl24_world"].__dict__,
            },
            "coordinate_spaces": {
                "gt_keypoints": "new_world",
                "gt_keypoints_world": "world",
                "mhr_keypoints_world": "world",
                "mhr_vertices_world": "world",
                "gt_vertices_world": "world",
                "gt_vertices_new_world": "new_world",
                "gt_keypoints_lidar": "lidar_sensor",
                "gt_keypoints_pc_centered_input_lidar": "pc_centered_input_lidar",
                "input_lidar": "lidar_sensor",
            },
            "cameras": {
                "rgb_camera_convention": "intrinsic_plus_extrinsic_world_to_camera_in_new_world",
                "lidar_camera_convention": "intrinsic_plus_extrinsic_world_to_camera_in_new_world",
                "lidar_pose_encoding_frame_hw": [1, int(input_lidar_sensor.shape[0])],
            },
            "conversion": {
                "selected_backend": smpl_fit["backend"],
                "fitting_error": float(smpl_fit["fitting_error"]),
                "edge_error": float(smpl_fit["edge_error"]),
                "backend_metadata": self.smpl_adapter.backend_metadata(),
            },
            "lidar_version": self.cfg.lidar_version,
            "artifacts": artifacts,
        }
        self._save_manifest(export_dir / "manifest.json", metadata)
        metadata["manifest_path"] = str(export_dir / "manifest.json")
        return metadata

    def _build_panoptic_export(
        self,
        sample_dir: Path,
        base_manifest: dict[str, Any],
        sam3d_output: dict[str, Any],
    ) -> dict[str, Any]:
        image_size_hw = tuple(int(x) for x in base_manifest["image_size_hw"])
        mhr_keypoints_world = (
            np.asarray(sam3d_output["pred_keypoints_3d"], dtype=np.float32)
            + np.asarray(sam3d_output["pred_cam_t"], dtype=np.float32)[None, :]
        )
        panoptic_world = mhr70_to_panoptic19(mhr_keypoints_world)
        body_center = np.asarray(panoptic_world[2], dtype=np.float32)
        R_new_to_world = estimate_panoptic_root_rotation(panoptic_world)
        panoptic_new_world = world_to_new_world(panoptic_world, body_center, R_new_to_world)

        rgb_camera_world = build_rgb_camera_from_sam3d(
            image_size_hw=image_size_hw,
            focal_length=sam3d_output["focal_length"],
            extrinsic_world_to_camera=np.hstack([np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)]),
        )
        rgb_camera_new_world = {
            "intrinsic": rgb_camera_world["intrinsic"],
            "extrinsic": update_extrinsic_for_new_world(
                rgb_camera_world["extrinsic"], body_center, R_new_to_world
            ),
        }

        export_dir = sample_dir / self.cfg.export_dirname / "panoptic"
        export_dir.mkdir(parents=True, exist_ok=True)
        save_numpy(export_dir / "gt_keypoints.npy", panoptic_new_world)
        save_numpy(export_dir / "gt_keypoints_world.npy", panoptic_world)
        save_numpy(export_dir / "gt_pelvis.npy", body_center)
        save_camera_json(export_dir / "rgb_camera.json", rgb_camera_new_world)

        artifacts: dict[str, str] = {
            "gt_keypoints": str(export_dir / "gt_keypoints.npy"),
            "gt_keypoints_world": str(export_dir / "gt_keypoints_world.npy"),
            "gt_pelvis": str(export_dir / "gt_pelvis.npy"),
            "rgb_camera": str(export_dir / "rgb_camera.json"),
        }
        if self.cfg.save_pose_encodings:
            gt_camera_rgb = camera_to_pose_encoding(rgb_camera_new_world, image_size_hw)
            save_numpy(export_dir / "gt_camera_rgb.npy", gt_camera_rgb)
            artifacts["gt_camera_rgb"] = str(export_dir / "gt_camera_rgb.npy")
        if self.cfg.save_rgb_2d_keypoints:
            gt_keypoints_2d_rgb = normalize_points_2d(
                project_points_to_image(panoptic_world, rgb_camera_world),
                image_size_hw=image_size_hw,
            )
            save_numpy(export_dir / "gt_keypoints_2d_rgb.npy", gt_keypoints_2d_rgb)
            artifacts["gt_keypoints_2d_rgb"] = str(export_dir / "gt_keypoints_2d_rgb.npy")

        metadata = {
            "format_name": "panoptic_joints19",
            "status": "accepted",
            "topology": {
                "gt_keypoints": TOPOLOGY_METADATA["panoptic19_new_world"].__dict__,
                "gt_keypoints_world": TOPOLOGY_METADATA["panoptic19_world"].__dict__,
            },
            "coordinate_spaces": {
                "gt_keypoints": "new_world",
                "gt_keypoints_world": "world",
            },
            "notes": {
                "smpl_payload": "not_exported_for_panoptic",
            },
            "lidar_version": self.cfg.lidar_version,
            "artifacts": artifacts,
        }
        self._save_manifest(export_dir / "manifest.json", metadata)
        metadata["manifest_path"] = str(export_dir / "manifest.json")
        return metadata

    def export_sample(self, sample_dir: Path) -> dict[str, Any]:
        sample_dir = Path(sample_dir).expanduser().resolve()
        manifest = self._load_manifest(sample_dir)
        export_root = sample_dir / self.cfg.export_dirname
        export_root.mkdir(parents=True, exist_ok=True)

        export_manifest: dict[str, Any] = {
            "status": "rejected",
            "sample_dir": str(sample_dir),
            "source_manifest_path": str(sample_dir / "manifest.json"),
            "formats": {},
            "rejection_reason": None,
        }
        try:
            sam3d_output = self._rerun_sam3d(sample_dir, manifest)
            export_manifest["sam3d_rerun"] = {
                "recovered_keys": sorted(k for k in sam3d_output.keys() if k != "faces"),
                "checkpoint_root": str(Path(self.cfg.checkpoint_root).expanduser().resolve()),
            }
            export_manifest["formats"]["humman"] = self._build_humman_export(sample_dir, manifest, sam3d_output)
            export_manifest["formats"]["panoptic"] = self._build_panoptic_export(sample_dir, manifest, sam3d_output)
            export_manifest["status"] = "accepted"
        except Exception as exc:
            export_manifest["rejection_reason"] = str(exc)

        self._save_manifest(export_root / "export_manifest.json", export_manifest)
        export_manifest["manifest_path"] = str(export_root / "export_manifest.json")
        return export_manifest
