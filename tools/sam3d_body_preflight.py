#!/usr/bin/env python3
import argparse
import importlib
import os
import sys
from pathlib import Path


REQUIRED_PATHS = [
    "model_config.yaml",
    "model.ckpt",
    "assets/mhr_model.pt",
]


def _fail(message: str) -> None:
    print(f"[SAM3D preflight] ERROR: {message}")
    raise SystemExit(1)


def _check_cuda() -> None:
    try:
        import torch
    except Exception as exc:
        _fail(f"failed to import torch: {exc}")

    if not torch.cuda.is_available():
        _fail("CUDA is not available. SAM-3D-Body preflight requires GPU runtime.")
    if torch.cuda.device_count() < 1:
        _fail("No CUDA devices detected. SAM-3D-Body preflight requires at least one GPU.")
    print(f"[SAM3D preflight] CUDA OK: {torch.cuda.device_count()} device(s) visible")


def _add_submodule_to_path(repo_root: Path) -> Path:
    submodule_root = repo_root / "third_party" / "sam-3d-body"
    if not submodule_root.exists():
        _fail(f"submodule path does not exist: {submodule_root}")
    sys.path.insert(0, str(submodule_root))
    return submodule_root


def _check_required_assets(checkpoint_root: Path) -> tuple[Path, Path]:
    cfg_path = checkpoint_root / "model_config.yaml"
    ckpt_path = checkpoint_root / "model.ckpt"
    mhr_path = checkpoint_root / "assets" / "mhr_model.pt"
    required_paths = [checkpoint_root / rel for rel in REQUIRED_PATHS]

    missing = [p for p in required_paths if not p.exists()]
    unreadable = [p for p in required_paths if p.exists() and not os.access(p, os.R_OK)]
    if missing:
        joined = ", ".join(str(p) for p in missing)
        _fail(f"missing required checkpoint asset(s): {joined}")
    if unreadable:
        joined = ", ".join(str(p) for p in unreadable)
        _fail(f"unreadable required checkpoint asset(s): {joined}")

    print("[SAM3D preflight] checkpoint assets OK")
    return ckpt_path, mhr_path


def _check_imports() -> None:
    modules = [
        "sam_3d_body",
        "sam_3d_body.build_models",
        "sam_3d_body.sam_3d_body_estimator",
    ]
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            _fail(f"failed to import `{module_name}`: {exc}")
    print("[SAM3D preflight] imports OK")


def _check_model_load(ckpt_path: Path, mhr_path: Path) -> None:
    try:
        from sam_3d_body import load_sam_3d_body
    except Exception as exc:
        _fail(f"failed to import load_sam_3d_body: {exc}")

    try:
        model, model_cfg = load_sam_3d_body(
            checkpoint_path=str(ckpt_path),
            device="cuda",
            mhr_path=str(mhr_path),
        )
    except Exception as exc:
        _fail(f"load_sam_3d_body failed: {exc}")

    device = getattr(model, "device", "unknown")
    print(f"[SAM3D preflight] model load OK (device={device})")
    del model
    del model_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM-3D-Body environment preflight")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("/opt/data/SAM_3dbody_checkpoints"),
        help="Path to checkpoint root containing assets/ directory",
    )
    parser.add_argument(
        "--skip-model-load",
        action="store_true",
        help="Skip loading model checkpoint (import and file checks still run)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    print(f"[SAM3D preflight] repo root: {repo_root}")
    _check_cuda()
    _add_submodule_to_path(repo_root)
    ckpt_path, mhr_path = _check_required_assets(args.checkpoint_root)
    _check_imports()
    if not args.skip_model_load:
        _check_model_load(ckpt_path, mhr_path)
    print("[SAM3D preflight] SUCCESS")


if __name__ == "__main__":
    main()
