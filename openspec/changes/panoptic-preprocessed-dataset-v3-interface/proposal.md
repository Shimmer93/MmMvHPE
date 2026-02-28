## Why

MMHPE now has preprocessed single-actor Panoptic Kinoptic data, but no dataset class that loads it with the same training-facing interface as `humman_dataset_v3`. This blocks direct reuse of existing training pipelines and configs for RGB/depth/GT loading on Panoptic without ad-hoc code paths.

## What Changes

- Add a new Panoptic preprocessed dataset class under `datasets/` that loads sequence-preserving outputs from `/opt/data/panoptic_kinoptic_single_actor_cropped` by default.
- Align dataset return contract and metadata keys with `humman_dataset_v3` so training code can switch datasets with minimal/no model-side changes.
- Support split-aware sequence selection and robust sequence filtering for partially available preprocessed sequences.
- Provide dataset/config documentation and runnable config examples for train/val/test in the existing config-driven workflow.
- Keep scope to Panoptic single-actor preprocessed format (RGB, depth, GT keypoints, cameras metadata). No multi-actor support in this change.

## Capabilities

### New Capabilities
- `panoptic-preprocessed-dataset-v3-interface`: Dataset capability to load preprocessed Panoptic Kinoptic samples with a `humman_dataset_v3`-compatible training interface.
- `panoptic-preprocessed-split-and-sequence-selection`: Configurable split/sequence selection behavior for Panoptic preprocessed data, including partial-data-safe filtering.

### Modified Capabilities
- None.

## Impact

- Affected code:
  - `datasets/` (new Panoptic dataset class and registration/wiring),
  - `configs/` (new or updated dataset config blocks),
  - `docs/` (usage and data contract documentation).
- Affected modalities/components:
  - RGB,
  - depth,
  - GT keypoints,
  - per-sequence cropped calibration metadata.
- Dependencies/APIs:
  - no new external dependency expected,
  - dataset API compatibility targeted at existing training pipeline interfaces used with `humman_dataset_v3`.
- Runtime/log impact:
  - enables config-driven Panoptic training/eval/predict runs using existing pipeline structure,
  - expected artifacts remain under `logs/` using existing experiment/version conventions.
