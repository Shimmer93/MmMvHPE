from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .virtual_lidar import VirtualLidarConfig, sample_self_occlusion_aware_pointcloud
from .visualization import save_lidar_comparison_figure


@dataclass(frozen=True)
class SyntheticLidarRegenerationConfig:
    synthetic_root: str
    lidar_version: str = "v1"
    depth_buffer_resolution: int = 720
    depth_buffer_visibility_tolerance: float = 0.03
    depth_buffer_margin_ratio: float = 0.05
    depth_buffer_candidate_factor: int = 48
    overwrite_existing: bool = False
    save_qc: bool = False
    qc_dirname: str = "lidar_qc"


class SyntheticLidarRegenerationPipeline:
    def __init__(self, cfg: SyntheticLidarRegenerationConfig) -> None:
        self.cfg = cfg
        self.synthetic_root = Path(cfg.synthetic_root).expanduser().resolve()
        if not self.synthetic_root.is_dir():
            raise FileNotFoundError(f"Synthetic root not found: {self.synthetic_root}")
        if str(cfg.lidar_version).strip().lower() != "v1":
            raise ValueError(
                f"SyntheticLidarRegenerationPipeline currently supports lidar_version='v1' only, "
                f"got {cfg.lidar_version}."
            )
        self.lidar_version = "v1"

    def list_sample_dirs(self) -> list[Path]:
        return sorted(
            p for p in self.synthetic_root.iterdir() if p.is_dir() and p.name.startswith("ann_")
        )

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _save_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _artifact_path(manifest: dict[str, Any], key: str) -> Path:
        artifacts = manifest.get("artifacts", {})
        if key not in artifacts:
            raise ValueError(f"Manifest is missing artifact key `{key}`.")
        return Path(artifacts[key]).expanduser().resolve()

    def _load_sample_assets(self, sample_dir: Path) -> dict[str, Any]:
        sample_dir = Path(sample_dir).expanduser().resolve()
        manifest_path = sample_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Sample manifest not found: {manifest_path}")
        manifest = self._load_json(manifest_path)
        if manifest.get("status") != "accepted":
            raise ValueError(f"Synthetic sample is not accepted: {sample_dir.name}")

        vertices_world = np.load(self._artifact_path(manifest, "pred_vertices_canonical")).astype(np.float32)
        faces = np.load(self._artifact_path(manifest, "mesh_faces")).astype(np.int32)
        extrinsic = np.load(self._artifact_path(manifest, "lidar_extrinsic_world_to_sensor")).astype(np.float32)
        pointcloud_v0a = np.load(self._artifact_path(manifest, "synthetic_lidar_points_sensor")).astype(np.float32)
        sensor_position = np.asarray(manifest["lidar_pose"]["sensor_position_world"], dtype=np.float32).reshape(3)
        return {
            "sample_dir": sample_dir,
            "manifest_path": manifest_path,
            "manifest": manifest,
            "vertices_world": vertices_world,
            "faces": faces,
            "extrinsic_world_to_sensor": extrinsic,
            "sensor_position_world": sensor_position,
            "pointcloud_v0a_sensor": pointcloud_v0a,
        }

    @staticmethod
    def _build_v1_path(sample_dir: Path) -> Path:
        return sample_dir / "arrays" / "synthetic_lidar_points_sensor_v1.npy"

    @staticmethod
    def _build_qc_path(sample_dir: Path, qc_dirname: str) -> Path:
        return sample_dir / qc_dirname / "lidar_v0a_vs_v1.png"

    def _build_sampling_cfg(self, manifest: dict[str, Any], *, resolution: int) -> VirtualLidarConfig:
        lidar_sampling = manifest.get("lidar_sampling", {})
        num_points = int(lidar_sampling.get("num_points", 2048))
        oversample_factor = int(lidar_sampling.get("oversample_factor", 8))
        surface_noise_std = float(lidar_sampling.get("surface_noise_std", 0.002))
        return VirtualLidarConfig(
            num_points=num_points,
            oversample_factor=oversample_factor,
            surface_noise_std=surface_noise_std,
            depth_buffer_resolution=int(resolution),
            depth_buffer_visibility_tolerance=float(self.cfg.depth_buffer_visibility_tolerance),
            depth_buffer_margin_ratio=float(self.cfg.depth_buffer_margin_ratio),
            depth_buffer_candidate_factor=int(self.cfg.depth_buffer_candidate_factor),
        )

    def generate_v1_pointcloud(
        self,
        sample_dir: Path,
        *,
        depth_buffer_resolution: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
        assets = self._load_sample_assets(sample_dir)
        manifest = assets["manifest"]
        resolution = (
            int(self.cfg.depth_buffer_resolution)
            if depth_buffer_resolution is None
            else int(depth_buffer_resolution)
        )
        lidar_cfg = self._build_sampling_cfg(manifest, resolution=resolution)
        seed = int(manifest.get("annotation_id", 0)) + resolution
        rng = random.Random(seed)
        start_time = time.perf_counter()
        pointcloud_v1, metadata = sample_self_occlusion_aware_pointcloud(
            vertices_world=assets["vertices_world"],
            faces=assets["faces"],
            sensor_position_world=assets["sensor_position_world"],
            extrinsic_world_to_sensor=assets["extrinsic_world_to_sensor"],
            cfg=lidar_cfg,
            rng=rng,
        )
        metadata = dict(metadata)
        metadata["runtime_sec"] = float(time.perf_counter() - start_time)
        metadata["random_seed"] = int(seed)
        return pointcloud_v1, metadata, assets

    @staticmethod
    def _ensure_v0a_entry(manifest: dict[str, Any]) -> dict[str, Any]:
        lidar_artifacts = dict(manifest.get("lidar_artifacts", {}))
        if "v0a" not in lidar_artifacts:
            artifacts = manifest.get("artifacts", {})
            lidar_sampling = manifest.get("lidar_sampling", {})
            lidar_artifacts["v0a"] = {
                "artifact_path": artifacts.get("synthetic_lidar_points_sensor"),
                "simulation_mode": manifest.get("simulation_mode", "visible_surface_sampling_v0a"),
                "num_points": int(lidar_sampling.get("num_points", 2048)),
                "oversample_factor": int(lidar_sampling.get("oversample_factor", 8)),
                "surface_noise_std": float(lidar_sampling.get("surface_noise_std", 0.002)),
            }
        return lidar_artifacts

    def regenerate_sample(self, sample_dir: Path) -> dict[str, Any]:
        sample_dir = Path(sample_dir).expanduser().resolve()
        v1_path = self._build_v1_path(sample_dir)
        if v1_path.is_file() and not self.cfg.overwrite_existing:
            return {
                "sample_dir": str(sample_dir),
                "status": "skipped_existing",
                "lidar_version": self.lidar_version,
                "artifact_path": str(v1_path),
            }

        pointcloud_v1, metadata, assets = self.generate_v1_pointcloud(sample_dir)
        sample_dir.joinpath("arrays").mkdir(parents=True, exist_ok=True)
        np.save(v1_path, pointcloud_v1.astype(np.float32))

        manifest = dict(assets["manifest"])
        artifacts = dict(manifest.get("artifacts", {}))
        artifacts["synthetic_lidar_points_sensor_v1"] = str(v1_path)
        manifest["artifacts"] = artifacts

        lidar_artifacts = self._ensure_v0a_entry(manifest)
        lidar_artifacts["v1"] = {
            "artifact_path": str(v1_path),
            **metadata,
        }
        manifest["lidar_artifacts"] = lidar_artifacts
        manifest["available_lidar_versions"] = sorted(lidar_artifacts.keys())
        manifest.setdefault("default_lidar_version", "v0a")

        if self.cfg.save_qc:
            qc_path = self._build_qc_path(sample_dir, self.cfg.qc_dirname)
            save_lidar_comparison_figure(
                qc_path,
                canonical_vertices=assets["vertices_world"],
                sensor_position_world=assets["sensor_position_world"],
                pointclouds_sensor=[assets["pointcloud_v0a_sensor"], pointcloud_v1],
                titles=["LiDAR v0-a", f"LiDAR {self.lidar_version}"],
            )
            artifacts["synthetic_lidar_qc_v1"] = str(qc_path)

        self._save_json(assets["manifest_path"], manifest)
        return {
            "sample_dir": str(sample_dir),
            "status": "accepted",
            "lidar_version": self.lidar_version,
            "artifact_path": str(v1_path),
            "metadata": metadata,
            "qc_path": artifacts.get("synthetic_lidar_qc_v1"),
        }
