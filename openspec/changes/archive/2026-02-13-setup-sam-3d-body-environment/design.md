## Context

MMHPE now vendors `third_party/sam-3d-body` as a submodule, but there is no validated environment contract yet for running it in the repository's `uv` workflow. The repo targets Python 3.12 and CUDA 12.4-sensitive dependencies, so uncontrolled dependency drift can break existing training paths (`main.py`, `datasets/`, `models/`) even if SAM-3D-Body is not used.

This change is environment-scoped: make SAM-3D-Body runnable in the same project environment, define the checkpoint path contract at `/opt/data/SAM_3dbody_checkpoints/`, and provide smoke validation. No training/inference integration into `models/model_api.py` or config pipelines is included yet.

## Goals / Non-Goals

**Goals:**
- Define how SAM-3D-Body dependencies are installed and executed under `uv`.
- Preserve existing MMHPE runtime behavior for RGB/depth/LiDAR/mmWave training and evaluation.
- Add a repeatable smoke check path to validate submodule import/init and checkpoint discovery.
- Document setup and troubleshooting in `docs/` with concrete commands.
- Treat SAM-3D-Body as a required environment component for this repository.

**Non-Goals:**
- No benchmark implementation or comparison pipeline in `main.py`.
- No dataset adapters or new configs under `configs/` for SAM-3D-Body inference.
- No changes to `uv.lock` (per project policy).
- No changes to existing model architectures or trainer logic in `models/model_api.py`.

## Decisions

1. Keep setup isolated from main training pipeline.
Why: reduces regression risk to current MMHPE runs.
Alternative considered: directly wiring SAM-3D-Body into `main.py` now. Rejected because environment validation should be proven first.

2. Use documentation-first setup plus lightweight validation script.
Why: this phase is about reproducibility and operability; a small script under `tools/` can check imports, version assumptions, CUDA visibility, and checkpoint path existence.
Alternative considered: only README instructions without executable checks. Rejected because it is harder to verify in CI/remote machines.

3. Make SAM-3D-Body dependencies non-optional in project environment setup.
Why: the target workflow requires SAM-3D-Body availability by default for upcoming benchmark/demo work.
Alternative considered: keep SAM-3D-Body optional. Rejected per project direction.

4. Standardize checkpoint location and required asset files as external data contract.
Why: `/opt/data/SAM_3dbody_checkpoints/` is already provisioned and should remain outside repo.
Alternative considered: tracking checkpoints under repo paths. Rejected due to repository size and portability concerns.

Required files:
- `/opt/data/SAM_3dbody_checkpoints/model_config.yaml`
- `/opt/data/SAM_3dbody_checkpoints/model.ckpt`
- `/opt/data/SAM_3dbody_checkpoints/assets/mhr_model.pt`

## Risks / Trade-offs

- [Dependency conflicts with MMHPE CUDA stack] -> Mitigation: isolate optional deps, validate with smoke script, and document compatible versions.
- [Submodule requires package versions incompatible with Python 3.12] -> Mitigation: document constraints and pin compatible versions in setup guidance; fail setup early with explicit errors.
- [Users assume SAM-3D-Body is integrated into training] -> Mitigation: explicitly document scope boundaries and deferred integration tasks.
- [Checkpoint assets missing or permission issues] -> Mitigation: add explicit preflight checks for required files and actionable error messages in validation script/docs.

## Migration Plan

1. Add environment setup documentation for SAM-3D-Body under `docs/` with `uv` commands.
2. Add a smoke-check script in `tools/` that validates:
   - submodule import path,
   - required package importability,
   - CUDA visibility assumptions,
   - required checkpoint assets exist and are readable:
     - `/opt/data/SAM_3dbody_checkpoints/model_config.yaml`
     - `/opt/data/SAM_3dbody_checkpoints/model.ckpt`
     - `/opt/data/SAM_3dbody_checkpoints/assets/mhr_model.pt`
3. Add minimal `pyproject.toml` adjustments only if required by the smoke check.
4. Run smoke check locally and record expected output in docs.
5. Rollback strategy: remove new optional setup entries and tool script; no core training path rollback needed because no core path is modified.

## Open Questions

- None for environment scope after locking required dependencies, GPU requirement, and checkpoint asset filenames.
