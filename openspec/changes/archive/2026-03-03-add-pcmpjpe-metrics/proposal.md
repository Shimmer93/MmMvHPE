## Why

Current evaluation reports MPJPE without explicitly compensating pelvis translation and pelvis orientation mismatch between prediction and ground truth. For MMHPE pipelines that output either SMPL joints or direct keypoints, this can overstate structural joint error when global root pose differs, making cross-model comparisons less diagnostic. We need a consistent pelvis-centered, root-rotation-aligned metric that can be enabled in existing RGB+PC workflows now.

## What Changes

- Add a new non-SMPL metric `PCMPJPE` in `metrics/mpjpe.py`.
- Add a new SMPL metric `SMPL_PCMPJPE` in `metrics/smpl_metrics.py`.
- Define metric behavior to align pelvis coordinates between prediction and GT, then remove pelvis root rotation before MPJPE computation.
- For non-SMPL keypoints, derive pelvis orientation with the same convention used by `remove_root_rotation` in `datasets/panoptic_preprocessed_dataset_v1.py`.
- Expose the new metrics in relevant config files used by the current RGB+PC pipeline.
- Add tests for metric correctness and config integration.

Scope boundaries / non-goals:
- No changes to model architectures, losses, or training targets.
- No changes to dataset content/format; only metric-time alignment logic is added.
- No broad fallback behaviors for malformed inputs; invalid assumptions should raise explicit errors.

## Capabilities

### New Capabilities
- `pelvis-centered-pose-metrics`: Evaluate pose error after pelvis translation and pelvis orientation alignment for both SMPL and direct-keypoint outputs.

### Modified Capabilities
- None.

## Impact

- Affected code: `metrics/mpjpe.py`, `metrics/smpl_metrics.py`, and metric-related tests.
- Affected configs: RGB+PC training/evaluation configs that currently report MPJPE-family metrics.
- Runtime/log impact: additional scalar metric entries (`PCMPJPE`, `SMPL_PCMPJPE`) in evaluation outputs and logs under `logs/`.
- Dependencies/APIs: no new external dependencies expected; existing metric interfaces are extended.
