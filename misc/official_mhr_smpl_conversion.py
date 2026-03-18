from __future__ import annotations

import contextlib
import ctypes
import importlib
import os
import sys
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MHR_ROOT = REPO_ROOT / "third_party" / "MHR"
DEFAULT_SMPL_MODEL_PATH = Path("/opt/data/SMPL_NEUTRAL.pkl")


@contextlib.contextmanager
def _pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class OfficialSam3dToSmplConverter:
    def __init__(
        self,
        *,
        device: str = "cuda",
        mhr_root: str | Path = DEFAULT_MHR_ROOT,
        smpl_model_path: str | Path = DEFAULT_SMPL_MODEL_PATH,
        batch_size: int = 128,
    ) -> None:
        self.device = str(device)
        self.batch_size = int(batch_size)
        self.mhr_root = Path(mhr_root).expanduser().resolve()
        self.smpl_model_path = Path(smpl_model_path).expanduser().resolve()
        self.tool_dir = self.mhr_root / "tools" / "mhr_smpl_conversion"
        self.tool_assets_dir = self.tool_dir / "assets"
        self.mhr_assets_dir = self.mhr_root / "assets"
        self._validate_environment()
        self._setup_import_paths()
        self._load_components()

    def _validate_environment(self) -> None:
        if not self.mhr_root.exists():
            raise FileNotFoundError(
                f"Official MHR conversion requires a checkout at {self.mhr_root}, but it does not exist."
            )
        if not self.tool_dir.exists():
            raise FileNotFoundError(
                f"Official MHR conversion tools are missing under {self.tool_dir}."
            )
        required_mhr_assets = [
            self.mhr_assets_dir / "compact_v6_1.model",
            self.mhr_assets_dir / "lod1.fbx",
            self.mhr_assets_dir / "corrective_blendshapes_lod1.npz",
            self.mhr_assets_dir / "corrective_activation.npz",
        ]
        required_tool_assets = [
            self.tool_assets_dir / "head_hand_mask.npz",
            self.tool_assets_dir / "mhr_face_mask.ply",
            self.tool_assets_dir / "subsampled_vertex_indices.npy",
            self.tool_assets_dir / "mhr2smpl_mapping.npz",
        ]
        required = required_mhr_assets + required_tool_assets + [self.smpl_model_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Official MHR-to-SMPL conversion assets are missing: " + ", ".join(missing)
            )
        try:
            self._prepare_torch_runtime_for_pymomentum()
            importlib.import_module("smplx")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Official MHR conversion requires the `smplx` package.") from exc
        try:
            importlib.import_module("torch")
            importlib.import_module("pymomentum.geometry")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Official MHR conversion requires Meta's `pymomentum` package with "
                "`pymomentum.geometry`. The generic PyPI "
                "`pymomentum` package is not sufficient."
            ) from exc

    def _prepare_torch_runtime_for_pymomentum(self) -> None:
        torch_mod = importlib.import_module("torch")
        torch_lib_dir = Path(torch_mod.__file__).resolve().parent / "lib"
        if not torch_lib_dir.exists():
            return
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [str(torch_lib_dir)]
        if existing:
            parts.append(existing)
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
        for lib_name in ("libc10.so", "libtorch_cpu.so", "libtorch.so", "libtorch_python.so"):
            lib_path = torch_lib_dir / lib_name
            if lib_path.exists():
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)

    def _setup_import_paths(self) -> None:
        for path in (self.mhr_root, self.tool_dir):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

    def _load_components(self) -> None:
        import smplx
        import pymomentum.geometry as pym_geometry
        from conversion import Conversion

        self._smplx = smplx
        self._pym_geometry = pym_geometry
        self._Conversion = Conversion
        with _pushd(self.tool_dir):
            fbx_path = self.mhr_assets_dir / "lod1.fbx"
            model_path = self.mhr_assets_dir / "compact_v6_1.model"
            character = self._pym_geometry.Character.load_fbx(
                str(fbx_path),
                str(model_path),
                load_blendshapes=False,
            )
            self._mhr_model = _MinimalOfficialMhrModel(character)
            self._smpl_model = self._smplx.SMPL(model_path=str(self.smpl_model_path)).to(self.device)
            self._converter = self._Conversion(
                mhr_model=self._mhr_model,
                smpl_model=self._smpl_model,
                method="pytorch",
            )
            self._converter._DEVICE = self.device
            self._converter._smpl_model = self._smpl_model.to(self.device)

    def convert_outputs(
        self,
        sam3d_outputs: list[dict[str, Any]],
        *,
        batch_size: int | None = None,
        return_vertices: bool = False,
        return_errors: bool = True,
    ) -> dict[str, Any]:
        if not sam3d_outputs:
            raise ValueError("Official SAM3D-to-SMPL conversion requires at least one SAM3D output.")
        effective_batch_size = int(batch_size or self.batch_size)
        mhr_vertices = []
        for sam3d_output in sam3d_outputs:
            pred_vertices = np.asarray(sam3d_output["pred_vertices"], dtype=np.float32)
            pred_cam_t = np.asarray(sam3d_output["pred_cam_t"], dtype=np.float32).reshape(1, 3)
            mhr_vertices.append(100.0 * (pred_vertices + pred_cam_t))
        mhr_vertices = np.stack(mhr_vertices, axis=0)
        with _pushd(self.tool_dir):
            result = self._converter.convert_mhr2smpl(
                mhr_vertices=mhr_vertices,
                single_identity=False,
                is_tracking=False,
                return_smpl_meshes=False,
                return_smpl_parameters=True,
                return_smpl_vertices=return_vertices,
                return_fitting_errors=return_errors,
                batch_size=effective_batch_size,
            )
        if result.result_parameters is None:
            raise RuntimeError("Official MHR conversion did not return SMPL parameters.")

        smpl_parameters = result.result_parameters
        batched_params = {}
        for key, value in smpl_parameters.items():
            if isinstance(value, torch.Tensor):
                batched_params[key] = value.to(self.device)
            else:
                batched_params[key] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            smpl_output = self._smpl_model(**batched_params)
        joints = smpl_output.joints.detach().cpu().numpy().astype(np.float32)
        if joints.shape[1] < 24:
            raise ValueError(f"Expected at least 24 SMPL joints, got shape {joints.shape}.")
        return {
            "smpl_joints24": joints[:, :24, :].copy(),
            "smpl_vertices": None if result.result_vertices is None else np.asarray(result.result_vertices, dtype=np.float32),
            "smpl_parameters": smpl_parameters,
            "fitting_errors": None if result.result_errors is None else np.asarray(result.result_errors, dtype=np.float32),
        }


class _MinimalOfficialMhrModel:
    def __init__(self, character: Any) -> None:
        self.character = character

    def to(self, device: str | torch.device) -> "_MinimalOfficialMhrModel":
        return self
