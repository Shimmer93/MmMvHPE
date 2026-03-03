## Why

We need a controlled Panoptic benchmark to compare `PanopticHPE` (seq3) and `XFi` under the same RGB+LiDAR data regime. The current Panoptic dataset loader does not expose HuMMan-style anchor-coordinate behavior, which blocks the requested XFi setup with GT keypoints anchored in RGB camera coordinates.

## What Changes

- Add optional anchor-coordinate output to `PanopticPreprocessedDatasetV1` so GT keypoints (and camera extrinsics) can be expressed in the selected anchor sensor frame, matching HuMMan dataset behavior.
- Keep default Panoptic behavior unchanged when anchor mode is not enabled.
- Add/verify support for foreground-only depth loading in Panoptic configs for RGB+LiDAR training.
- Create two Panoptic training configs for model comparison:
  - `PanopticHPE` seq3 on temporal split, 1 RGB + 1 LiDAR, foreground depth, excluding piano2/3/4.
  - `XFi` on the same Panoptic setup with GT anchored to RGB camera coordinates.
- Create four Panoptic test configs:
  - PanopticHPE on occluded sequence subset.
  - PanopticHPE on unoccluded subset.
  - XFi on occluded sequence subset.
  - XFi on unoccluded subset.

## Capabilities

### New Capabilities
- `panoptic-anchor-coordinate-output`: Optional anchor-coordinate conversion in Panoptic preprocessed dataset samples (camera extrinsics + GT keypoints).
- `panoptic-model-comparison-configs`: Panoptic RGB+LiDAR temporal-split config set for PanopticHPE vs XFi training/testing on occluded and unoccluded subsets.

### Modified Capabilities
- `panoptic-kinoptic-sequence-synchronized-crop-format`: Extend loader-facing behavior for preprocessed Panoptic outputs to support optional foreground-depth consumption and anchor-coordinate training semantics.

## Impact

- Affected code: `datasets/panoptic_preprocessed_dataset_v1.py`, new/updated YAML files under `configs/`.
- Affected docs: Panoptic preprocessing/dataset usage docs in `docs/`.
- Runtime impact: additional dataset-mode branch for anchor transforms; new experiment configs produce separate logs/checkpoints for model comparison runs.
