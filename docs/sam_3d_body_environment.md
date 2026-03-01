# SAM-3D-Body Environment Setup

This document defines the required environment setup to run the `third_party/sam-3d-body` submodule inside MMHPE.

## Scope

- This setup is required for this repository.
- GPU runtime is required.
- This is environment setup and preflight validation only; benchmark/demo integration into MMHPE training pipelines is handled in follow-up changes.

## Required Checkpoints

Checkpoint root must be:

- `/opt/data/SAM_3dbody_checkpoints/`

Required files:

- `/opt/data/SAM_3dbody_checkpoints/model_config.yaml`
- `/opt/data/SAM_3dbody_checkpoints/model.ckpt`
- `/opt/data/SAM_3dbody_checkpoints/mhr_model.pt`

## Install / Sync

Use the repository environment:

```bash
uv sync --frozen
```

If `uv sync --frozen` reports that the lockfile is out of date, run:

```bash
uv sync
```

Interim bootstrap for this environment (if preflight import checks fail):

```bash
uv pip install omegaconf yacs braceexpand roma pyrootutils pyrender trimesh
```

## Preflight Validation

Run the SAM-3D-Body preflight:

```bash
uv run --no-sync python tools/sam3d_body_preflight.py \
  --checkpoint-root /opt/data/SAM_3dbody_checkpoints
```

Optional faster check (skip checkpoint deserialization and model init):

```bash
uv run --no-sync python tools/sam3d_body_preflight.py \
  --checkpoint-root /opt/data/SAM_3dbody_checkpoints \
  --skip-model-load
```

## Baseline MMHPE Sanity Check

After preflight passes, run a known-good MMHPE command to verify baseline startup still works:

```bash
uv run --no-sync python main.py -c configs/dev/humman_fusionformer_gt_pose_mini.yml -g 1 -n 1 -w 2 -b 1 -e 1 -d 1
```

## Troubleshooting

- Import errors like `No module named 'sam_3d_body'`:
  - Ensure submodule exists at `third_party/sam-3d-body`.
  - Re-run `uv sync` and retry preflight.

- Import errors for dependency modules (for example `omegaconf`, `yacs`, `roma`, `braceexpand`):
  - Ensure environment sync completed successfully.
  - Re-run preflight to confirm.

- Missing checkpoint assets:
  - Ensure all three files exist under `/opt/data/SAM_3dbody_checkpoints/`.
  - Preflight prints the exact missing file path.

- CUDA errors:
  - Verify CUDA devices are visible in the active environment.
  - Preflight fails if `torch.cuda.is_available()` is false.
