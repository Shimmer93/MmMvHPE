## Why

Rerun visualization for inference is currently difficult to extend and inconsistent to operate, especially when adding SAM-3D-Body outputs on top of existing MMHPE visualization flows. We need a config-driven, reusable rerun pipeline now so SAM-3D-Body demo and benchmark inspections can be run in a predictable way on HuMMan data.

Current `scripts/visualize_inference_rerun.py` behavior also assumes fixed input modality/views in the display layout, which does not match multi-view/per-modality settings in configs.

## What Changes

- Refactor the current rerun inference visualization flow into a shared, model-agnostic visualization pipeline that loads dataset/config context from config files.
- Make input view display fully config-driven (modalities and number of views per modality from dataset config), replacing fixed hardcoded layout assumptions.
- Add a SAM-3D-Body inference adapter that runs inference on dataset-selected RGB frames and logs outputs to rerun using the shared pipeline.
- Keep compatibility with existing rerun usage by preserving current behavior in `scripts/visualize_inference_rerun.py` through shared utilities and stable CLI patterns where feasible.
- Add clear docs and runnable command examples for both existing MMHPE inference visualization and SAM-3D-Body rerun visualization.
- Add output conventions under `logs/` for saved rerun traces/artifacts and metadata for reproducibility.

## Capabilities

### New Capabilities
- `rerun-visualization-pipeline`: Shared, config-driven rerun visualization core for loading dataset samples, running inference adapters, and logging synchronized 2D/3D outputs.
- `sam3d-body-rerun-inference`: SAM-3D-Body inference + rerun visualization workflow on MMHPE datasets (initially HuMMan-focused) using repository checkpoint contracts.

### Modified Capabilities

## Impact

- Affected code: `scripts/visualize_inference_rerun.py` (refactor to shared core), new/updated script(s) under `scripts/`, and shared visualization helpers under `misc/` or `tools/` as needed.
- Affected datasets/modalities: HuMMan RGB frames first; future extension may include depth/LiDAR/mmWave overlays but not required in this change.
- Affected runtime outputs: new rerun traces and optional metadata files in `logs/` for reproducible visual debugging.
- Dependency/system impact: leverages existing SAM-3D-Body environment setup and checkpoint contract (`/opt/data/SAM_3dbody_checkpoints`).
- Scope boundary: this change does not add training/evaluation metrics or alter model training pipelines in `main.py`.
