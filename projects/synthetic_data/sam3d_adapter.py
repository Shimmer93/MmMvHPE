from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class SAM3DPaths:
    checkpoint_root: Path
    checkpoint_path: Path
    mhr_path: Path
    config_path: Path


def validate_sam3d_paths(checkpoint_root: str | Path) -> SAM3DPaths:
    root = Path(checkpoint_root).expanduser().resolve()
    config_path = root / "model_config.yaml"
    checkpoint_path = root / "model.ckpt"
    mhr_path = root / "assets" / "mhr_model.pt"
    missing = [p for p in [config_path, checkpoint_path, mhr_path] if not p.exists()]
    if missing:
        joined = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing required SAM-3D-Body checkpoint files: {joined}")
    return SAM3DPaths(
        checkpoint_root=root,
        checkpoint_path=checkpoint_path,
        mhr_path=mhr_path,
        config_path=config_path,
    )


class SAM3DRunner:
    def __init__(self, repo_root: str | Path, checkpoint_root: str | Path, device: str = "cuda") -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.paths = validate_sam3d_paths(checkpoint_root)
        self.device = str(device)
        if self.device != "cuda":
            raise ValueError(f"v0-a expects CUDA runtime for SAM-3D-Body, got device={self.device}.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. SAM-3D-Body requires GPU runtime.")

        submodule_root = self.repo_root / "third_party" / "sam-3d-body"
        if not submodule_root.exists():
            raise FileNotFoundError(f"SAM-3D-Body submodule not found: {submodule_root}")
        if str(submodule_root) not in sys.path:
            sys.path.insert(0, str(submodule_root))

        # Keep SAM-3D-Body on its TorchScript MHR path even if the official
        # MHR conversion runtime is installed in the environment.
        os.environ["MOMENTUM_ENABLED"] = "0"

        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

        model, model_cfg = load_sam_3d_body(
            checkpoint_path=str(self.paths.checkpoint_path),
            device=self.device,
            mhr_path=str(self.paths.mhr_path),
        )
        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )
        self.faces = np.asarray(self.estimator.faces, dtype=np.int32)

    def infer(self, image_rgb: np.ndarray, bbox_xyxy: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected image_rgb shape (H,W,3), got {image_rgb.shape}")
        if bbox_xyxy.shape != (4,):
            raise ValueError(f"Expected bbox shape (4,), got {bbox_xyxy.shape}")
        if mask.shape != image_rgb.shape[:2]:
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {image_rgb.shape[:2]}")
        outputs = self.estimator.process_one_image(
            image_rgb,
            bboxes=bbox_xyxy.astype(np.float32),
            masks=mask.astype(np.uint8),
            use_mask=True,
        )
        if len(outputs) != 1:
            raise RuntimeError(
                f"Expected exactly one SAM-3D-Body output for explicit bbox+mask, got {len(outputs)}."
            )
        output = outputs[0]
        output["faces"] = self.faces.copy()
        return output

    def render_overlay(self, image_rgb: np.ndarray, output: dict[str, Any]) -> np.ndarray:
        from sam_3d_body.visualization.renderer import Renderer

        image_bgr = image_rgb[..., ::-1].copy()
        renderer = Renderer(
            focal_length=output["focal_length"],
            faces=self.faces,
        )
        rendered = renderer(
            output["pred_vertices"],
            output["pred_cam_t"],
            image_bgr,
            mesh_base_color=(0.65098039, 0.74117647, 0.85882353),
            scene_bg_color=(1, 1, 1),
        )
        rendered = np.clip(rendered * 255.0, 0.0, 255.0).astype(np.uint8)
        return rendered[..., ::-1]
