## Why

MMHPE now includes `third_party/sam-3d-body`, but the current project environment does not yet guarantee that this submodule can be imported and executed reliably on the target CUDA/Linux setup. We need a reproducible environment setup now so SAM-3D-Body can be used later for demo comparisons and benchmark runs on our datasets.

## What Changes

- Define and document the environment requirements to run `third_party/sam-3d-body` inside the MMHPE workflow (`uv`-based, Python 3.12+, CUDA 12.4 target).
- Add setup instructions and validation steps to ensure SAM-3D-Body can load with its required dependencies in this repo.
- Configure checkpoint path usage for SAM-3D-Body with project-standard expectations, using `/opt/data/SAM_3dbody_checkpoints/`.
- Add a minimal runtime verification path (environment-level smoke checks), without introducing model training or benchmark logic in this change.
- Record integration boundaries so this change does not modify training/evaluation behavior for existing MMHPE RGB/depth/LiDAR/mmWave pipelines.

## Capabilities

### New Capabilities
- `sam-3d-body-environment-setup`: Reproducible environment and runtime setup for using `third_party/sam-3d-body` within MMHPE, including dependency setup guidance, checkpoint path contract, and smoke-test verification.

### Modified Capabilities

## Impact

- Affected code and docs: environment/dependency config (`pyproject.toml` if needed), setup docs under `docs/`, and optional helper scripts for environment validation.
- External dependency impact: enables a new third-party submodule runtime (`third_party/sam-3d-body`) under MMHPE environment constraints.
- Runtime/output impact: no direct training/inference behavior changes yet; no new benchmark outputs in `logs/` in this change beyond optional setup verification logs.
- Scope boundary: dataset benchmarking, comparison demos, and SAM-3D-Body task-level integration with MMHPE models are explicitly deferred to follow-up changes.
