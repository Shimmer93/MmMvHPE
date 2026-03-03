## Why

The current `random_split` in Panoptic behaves like a sequence-level split, which is confusing and can exclude a target sequence entirely from `val/test`. We need split semantics that are explicit: use a dedicated `temporal_split` for per-sequence frame partitioning and keep sequence-level partitioning as an explicit list-based split.

## What Changes

- Introduce a new Panoptic split mode `temporal_split` for deterministic per-sequence temporal splitting: first ratio portion for train and remaining portion for val/test (for example, first 80% train and last 20% test).
- Move current sequence-level partition behavior (currently represented by random sequence assignment) to explicit sequence lists under `cross_subject_split` in `configs/datasets/panoptic_split_config.yml` (train/val sequence lists are written directly in config).
- Add clear inline comments in `configs/datasets/panoptic_split_config.yml` documenting the meaning of each split mode so users do not interpret `random_split` as random sequence partitioning.
- Apply the split behavior consistently across Panoptic modalities that share synchronized frame IDs (`rgb`, `depth`, and derived `lidar` from depth).
- Add clear validation/error behavior for invalid split ratios and sequences too short for the requested temporal split.
- Update dataset config documentation/examples so users can opt into the new split mode without changing preprocessing outputs.

## Capabilities

### New Capabilities
- `panoptic-sequence-temporal-split`: Support deterministic per-sequence temporal train/val split for Panoptic preprocessed datasets.

### Modified Capabilities
- None.

## Impact

- Affected code: `datasets/panoptic_preprocessed_dataset_v1.py` split resolution and dataset indexing logic.
- Affected configs: `configs/datasets/panoptic_split_config.yml` contract and Panoptic experiment/visualization configs that use `split_to_use`.
- Migration impact:
  - configs that need per-sequence head/tail splitting switch to `split_to_use: temporal_split`;
  - configs that need sequence-level partition use explicit `cross_subject_split` sequence lists.
- Affected workflow outputs: dataset sample composition and frame coverage in training/validation runs under `logs/`.
- Modalities: `rgb`, `depth`, and depth-derived `lidar` (no direct change to `mmWave`).
- Non-goals: no changes to preprocessing file layout, camera calibration handling, or model architecture/loss definitions.
