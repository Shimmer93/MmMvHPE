## 1. Environment and Dependency Setup

- [x] 1.1 Audit `third_party/sam-3d-body` runtime requirements and map required Python/CUDA packages against current `pyproject.toml`.
- [x] 1.2 Update `pyproject.toml` with required SAM-3D-Body dependencies (non-optional) while preserving MMHPE CUDA 12.4 compatibility assumptions.
- [x] 1.3 Run `uv sync` and confirm dependency resolution/install succeeds on the target machine.

## 2. Runtime Validation Tooling

- [x] 2.1 Add a SAM-3D-Body preflight script under `tools/` to validate submodule importability and GPU runtime availability.
- [x] 2.2 Implement required checkpoint asset checks in the preflight script.
- [x] 2.3 Validate `/opt/data/SAM_3dbody_checkpoints/model_config.yaml`, `/opt/data/SAM_3dbody_checkpoints/model.ckpt`, and `/opt/data/SAM_3dbody_checkpoints/assets/mhr_model.pt` with explicit per-file errors.
- [x] 2.4 Add a runnable command example for the preflight script using `uv run`.

## 3. Documentation

- [x] 3.1 Add setup documentation under `docs/` describing required environment, dependency installation, and GPU requirement for SAM-3D-Body.
- [x] 3.2 Document the checkpoint contract rooted at `/opt/data/SAM_3dbody_checkpoints/` and required files under `assets/`.
- [x] 3.3 Document troubleshooting guidance for dependency/import failures and missing checkpoint assets with expected error messages.

## 4. Verification

- [x] 4.1 Execute the SAM-3D-Body preflight command via `uv run` and record pass/fail evidence locally.
- [x] 4.2 Re-run a minimal MMHPE command via `uv run python main.py ...` (existing known-good config) to confirm no regression in baseline pipeline startup.
