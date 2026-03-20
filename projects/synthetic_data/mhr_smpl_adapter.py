from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import smplx
import torch

from misc.official_mhr_smpl_conversion import OfficialSam3dToSmplConverter


@dataclass(frozen=True)
class MHRSMPLAdapterPaths:
    mhr_repo_root: Path
    mhr2smpl_mapping: Path
    smpl_model_path: Path


@dataclass(frozen=True)
class MHRSMPLFitConfig:
    mhr_repo_root: str | None = None
    smpl_model_path: str = "weights/smpl/SMPL_NEUTRAL.pkl"
    device: str = "cuda"
    backend: str = "official_preferred"
    num_iters_stage1: int = 150
    num_iters_stage2: int = 250
    lr_stage1: float = 0.05
    lr_stage2: float = 0.01
    vertex_weight: float = 1.0
    edge_weight: float = 0.2
    body_pose_reg: float = 1e-4
    betas_reg: float = 1e-3


def discover_mhr_repo_root(repo_root: str | Path, explicit_path: str | None) -> Path:
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser().resolve())
    repo_root = Path(repo_root).expanduser().resolve()
    candidates.extend(
        [
            repo_root / "third_party" / "MHR",
            repo_root / "third_party" / "mhr",
            Path("/tmp/MHR"),
        ]
    )
    for candidate in candidates:
        if (candidate / "tools" / "mhr_smpl_conversion" / "assets" / "mhr2smpl_mapping.npz").is_file():
            return candidate
    joined = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not find an MHR checkout with tools/mhr_smpl_conversion/assets/mhr2smpl_mapping.npz. "
        f"Checked: {joined}"
    )


def validate_mhr_smpl_paths(repo_root: str | Path, cfg: MHRSMPLFitConfig) -> MHRSMPLAdapterPaths:
    repo_root = Path(repo_root).expanduser().resolve()
    mhr_repo_root = discover_mhr_repo_root(repo_root, cfg.mhr_repo_root)
    mhr2smpl_mapping = mhr_repo_root / "tools" / "mhr_smpl_conversion" / "assets" / "mhr2smpl_mapping.npz"
    smpl_model_path = Path(cfg.smpl_model_path).expanduser()
    if not smpl_model_path.is_absolute():
        smpl_model_path = (repo_root / smpl_model_path).resolve()
    missing = [p for p in [mhr2smpl_mapping, smpl_model_path] if not p.exists()]
    if missing:
        joined = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing required MHR/SMPL export files: {joined}")
    return MHRSMPLAdapterPaths(
        mhr_repo_root=mhr_repo_root,
        mhr2smpl_mapping=mhr2smpl_mapping,
        smpl_model_path=smpl_model_path,
    )


def _to_numpy(value: Any, *, dtype: np.dtype = np.float32) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


class MHRSMPLAdapter:
    def __init__(self, repo_root: str | Path, cfg: MHRSMPLFitConfig) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.cfg = cfg
        self.paths = validate_mhr_smpl_paths(self.repo_root, cfg)
        self.device = torch.device(cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
        self.active_backend = "uninitialized"
        self._official_converter: OfficialSam3dToSmplConverter | None = None
        self.mapping = None
        self.triangle_ids = None
        self.baryc_coords = None
        self.smpl_model = None
        self.smpl_faces = None

        backend = str(cfg.backend).lower()
        if backend not in {"official", "official_preferred", "local"}:
            raise ValueError(f"Unsupported MHR->SMPL backend `{cfg.backend}`.")

        official_error: Exception | None = None
        if backend in {"official", "official_preferred"}:
            try:
                self._init_official()
                self.active_backend = "official_mhr_conversion_pytorch"
                return
            except Exception as exc:
                official_error = exc
                if backend == "official":
                    raise

        self._init_local()
        self.active_backend = "local_vertex_fit_with_mhr_mapping_assets"
        self._official_init_error = None if official_error is None else str(official_error)

    def _init_official(self) -> None:
        self._official_converter = OfficialSam3dToSmplConverter(
            device=str(self.device),
            mhr_root=self.paths.mhr_repo_root,
            smpl_model_path=self.paths.smpl_model_path,
        )

    def _init_local(self) -> None:
        self.mapping = np.load(self.paths.mhr2smpl_mapping)
        self.triangle_ids = np.asarray(self.mapping["triangle_ids"], dtype=np.int64)
        self.baryc_coords = np.asarray(self.mapping["baryc_coords"], dtype=np.float32)
        self.smpl_model = self._build_smpl_model(self.paths.smpl_model_path)
        self.smpl_faces = np.asarray(self.smpl_model.faces, dtype=np.int32)

    def _build_smpl_model(self, smpl_model_path: Path):
        try:
            model = smplx.SMPL(model_path=str(smpl_model_path), gender="neutral").to(self.device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load SMPL model from {smpl_model_path}. "
                "Use an official chumpy-free SMPL model file as described in the MHR conversion README."
            ) from exc
        model.eval()
        return model

    def barycentric_map_mhr_to_smpl(self, mhr_vertices_world: np.ndarray, mhr_faces: np.ndarray) -> np.ndarray:
        if self.triangle_ids is None or self.baryc_coords is None:
            raise RuntimeError("Local barycentric mapping is unavailable because the local backend is not initialized.")
        verts = np.asarray(mhr_vertices_world, dtype=np.float32)
        faces = np.asarray(mhr_faces, dtype=np.int64)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError(f"Expected mhr vertices shape (V,3), got {verts.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"Expected mhr faces shape (F,3), got {faces.shape}")
        triangles = verts[faces[self.triangle_ids]]
        smpl_vertices = np.sum(triangles * self.baryc_coords[:, :, None], axis=1)
        return smpl_vertices.astype(np.float32)

    @staticmethod
    def _compute_edge_vectors(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        edges = torch.cat(
            [
                faces[:, [0, 1]],
                faces[:, [1, 2]],
                faces[:, [2, 0]],
            ],
            dim=0,
        )
        edges = torch.unique(torch.sort(edges, dim=1).values, dim=0)
        return vertices[:, edges[:, 1], :] - vertices[:, edges[:, 0], :]

    def _fit_local_to_mhr_vertices(
        self,
        mhr_vertices_world: np.ndarray,
        mhr_faces: np.ndarray,
    ) -> dict[str, Any]:
        if self.smpl_model is None or self.smpl_faces is None:
            raise RuntimeError("Local SMPL model is unavailable because the local backend is not initialized.")
        target_smpl_vertices = self.barycentric_map_mhr_to_smpl(mhr_vertices_world, mhr_faces)
        target = torch.from_numpy(target_smpl_vertices).float().to(self.device)[None]
        smpl_faces_t = torch.from_numpy(self.smpl_faces.astype(np.int64)).to(self.device)
        target_edge_vectors = self._compute_edge_vectors(target, smpl_faces_t).detach()

        body_pose = torch.zeros((1, 69), dtype=torch.float32, device=self.device, requires_grad=True)
        betas = torch.zeros((1, 10), dtype=torch.float32, device=self.device, requires_grad=True)
        global_orient = torch.zeros((1, 3), dtype=torch.float32, device=self.device, requires_grad=True)
        transl_init = target.mean(dim=1)
        transl = transl_init.detach().clone().requires_grad_(True)

        def run_stage(parameters: list[torch.Tensor], num_iters: int, lr: float, optimize_body: bool) -> None:
            opt = torch.optim.Adam(parameters, lr=lr)
            for _ in range(num_iters):
                opt.zero_grad(set_to_none=True)
                output = self.smpl_model(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    betas=betas,
                    transl=transl,
                )
                pred_vertices = output.vertices
                pred_edge_vectors = self._compute_edge_vectors(pred_vertices, smpl_faces_t)
                vertex_loss = torch.mean((pred_vertices - target) ** 2)
                edge_loss = torch.mean((pred_edge_vectors - target_edge_vectors) ** 2)
                reg_loss = self.cfg.betas_reg * torch.mean(betas ** 2)
                if optimize_body:
                    reg_loss = reg_loss + self.cfg.body_pose_reg * torch.mean(body_pose ** 2)
                loss = self.cfg.vertex_weight * vertex_loss + self.cfg.edge_weight * edge_loss + reg_loss
                loss.backward()
                opt.step()

        run_stage(
            parameters=[global_orient, transl, betas],
            num_iters=self.cfg.num_iters_stage1,
            lr=self.cfg.lr_stage1,
            optimize_body=False,
        )
        run_stage(
            parameters=[global_orient, transl, betas, body_pose],
            num_iters=self.cfg.num_iters_stage2,
            lr=self.cfg.lr_stage2,
            optimize_body=True,
        )

        with torch.no_grad():
            output = self.smpl_model(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                transl=transl,
            )
            pred_vertices = output.vertices
            pred_edge_vectors = self._compute_edge_vectors(pred_vertices, smpl_faces_t)
            fitting_error = torch.sqrt(torch.mean((pred_vertices - target) ** 2)).item()
            edge_error = torch.sqrt(torch.mean((pred_edge_vectors - target_edge_vectors) ** 2)).item()
            joints24 = output.joints[0, :24].detach().cpu().numpy().astype(np.float32)
            vertices = pred_vertices[0].detach().cpu().numpy().astype(np.float32)
            global_orient_np = global_orient[0].detach().cpu().numpy().astype(np.float32)
            body_pose_np = body_pose[0].detach().cpu().numpy().astype(np.float32)
            betas_np = betas[0].detach().cpu().numpy().astype(np.float32)
            transl_np = transl[0].detach().cpu().numpy().astype(np.float32)
            pose72 = np.concatenate([global_orient_np, body_pose_np], axis=0).astype(np.float32)
            smpl_params82 = np.concatenate([pose72[:72], betas_np[:10]], axis=0).astype(np.float32)
        return {
            "backend": "local_vertex_fit_with_mhr_mapping_assets",
            "smpl_model_path": str(self.paths.smpl_model_path),
            "mhr_repo_root": str(self.paths.mhr_repo_root),
            "mhr2smpl_mapping": str(self.paths.mhr2smpl_mapping),
            "fitting_error": float(fitting_error),
            "edge_error": float(edge_error),
            "smpl_vertices_world": vertices,
            "smpl_joints24_world": joints24,
            "smpl_faces": self.smpl_faces.copy(),
            "global_orient": global_orient_np,
            "body_pose": body_pose_np,
            "betas": betas_np,
            "transl": transl_np,
            "gt_smpl_params": smpl_params82,
        }

    def _fit_official_from_sam3d_output(self, sam3d_output: dict[str, Any]) -> dict[str, Any]:
        if self._official_converter is None:
            raise RuntimeError("Official converter is unavailable.")
        converted = self._official_converter.convert_outputs(
            [sam3d_output],
            return_vertices=True,
            return_errors=True,
        )
        params = converted["smpl_parameters"]

        global_orient = _to_numpy(params.get("global_orient", params.get("root_orient")))[0]
        body_pose = _to_numpy(params["body_pose"])[0]
        betas = _to_numpy(params["betas"])[0]
        transl_value = params.get("transl", params.get("translation"))
        if transl_value is None:
            raise KeyError("Official conversion did not return `transl` or `translation`.")
        transl = _to_numpy(transl_value)[0]
        pose72 = np.concatenate([global_orient, body_pose], axis=0).astype(np.float32)
        smpl_params82 = np.concatenate([pose72[:72], betas[:10]], axis=0).astype(np.float32)
        fitting_errors = converted.get("fitting_errors")
        fitting_error = float(np.asarray(fitting_errors, dtype=np.float32).reshape(-1)[0]) if fitting_errors is not None else float("nan")
        return {
            "backend": "official_mhr_conversion_pytorch",
            "smpl_model_path": str(self.paths.smpl_model_path),
            "mhr_repo_root": str(self.paths.mhr_repo_root),
            "mhr2smpl_mapping": str(self.paths.mhr2smpl_mapping),
            "fitting_error": fitting_error,
            "edge_error": float("nan"),
            "smpl_vertices_world": np.asarray(converted["smpl_vertices"], dtype=np.float32)[0],
            "smpl_joints24_world": np.asarray(converted["smpl_joints24"], dtype=np.float32)[0],
            "smpl_faces": np.asarray(self._official_converter._smpl_model.faces, dtype=np.int32),
            "global_orient": global_orient.astype(np.float32),
            "body_pose": body_pose.astype(np.float32),
            "betas": betas.astype(np.float32),
            "transl": transl.astype(np.float32),
            "gt_smpl_params": smpl_params82,
        }

    def fit_smpl_to_sam3d_output(self, sam3d_output: dict[str, Any]) -> dict[str, Any]:
        if self.active_backend == "official_mhr_conversion_pytorch":
            return self._fit_official_from_sam3d_output(sam3d_output)
        mhr_vertices_world = (
            np.asarray(sam3d_output["pred_vertices"], dtype=np.float32)
            + np.asarray(sam3d_output["pred_cam_t"], dtype=np.float32)[None, :]
        )
        mhr_faces = np.asarray(sam3d_output["faces"], dtype=np.int32)
        return self._fit_local_to_mhr_vertices(mhr_vertices_world, mhr_faces)

    def backend_metadata(self) -> dict[str, Any]:
        payload = {
            "backend": self.active_backend,
            "mhr_repo_root": str(self.paths.mhr_repo_root),
            "mhr2smpl_mapping": str(self.paths.mhr2smpl_mapping),
            "smpl_model_path": str(self.paths.smpl_model_path),
            "device": str(self.device),
            "fit_config": json.loads(json.dumps(self.cfg.__dict__)),
        }
        if getattr(self, "_official_init_error", None):
            payload["official_init_error"] = self._official_init_error
        return payload
