## Why

The current HuMMan SAM3D evaluation and visualization path uses a heuristic MHR70-to-SMPL24 joint remap, which is sufficient for rough metric prototyping but produces visibly incorrect SMPL topology and can distort evaluation quality. MMHPE now needs the official MHR-to-SMPL conversion path so HuMMan-side SAM3D evaluation, comparison figures, and downstream analysis use fitted SMPL outputs instead of an approximate joint mapping.

## What Changes

- Add an official MHR-to-SMPL conversion path for HuMMan SAM3D evaluation and visualization, using the upstream MHR conversion workflow instead of the current heuristic joint remap.
- Integrate the conversion path into HuMMan SAM3D evaluation scripts so SMPL24 metrics are computed from fitted SMPL outputs derived from SAM3D results.
- Integrate the same conversion path into HuMMan SAM3D visualization/export scripts so GT SMPL24, raw SAM MHR output, and converted SMPL output can be compared consistently.
- Add explicit dependency and asset requirements for the official MHR conversion stack, including model/package availability checks and fail-fast error handling.
- Document scope boundaries and runtime outputs under `logs/` for converted HuMMan evaluation and visualization artifacts.
- **BREAKING**: HuMMan SAM3D evaluation/visualization outputs that currently rely on the heuristic MHR70-to-SMPL24 adapter will change to use fitted SMPL outputs, so metric values and visual outputs will not be directly comparable to older heuristic-adapter runs.

## Capabilities

### New Capabilities
- `sam3d-humman-official-smpl-conversion`: Convert SAM3D HuMMan outputs to fitted SMPL outputs using the official MHR conversion workflow for evaluation and visualization.

### Modified Capabilities
- `sam-3d-body-environment-setup`: Extend the environment contract to cover the additional MHR conversion dependencies and model assets required by the official conversion workflow.
- `sam3d-body-rerun-inference`: Extend HuMMan-side SAM3D visualization requirements so SMPL comparison outputs can use official converted SMPL results instead of heuristic joint remapping.

## Impact

Affected areas include HuMMan RGB evaluation and visualization scripts, shared SAM3D utility code, dependency/model setup, and documentation. The main runtime impact is on HuMMan test/evaluation outputs under `logs/`, where new converted-SMPL metrics and visual artifacts will replace heuristic-adapter outputs; Panoptic, depth, LiDAR, and mmWave pipelines are out of scope for this change unless they explicitly opt into the same conversion path later.
