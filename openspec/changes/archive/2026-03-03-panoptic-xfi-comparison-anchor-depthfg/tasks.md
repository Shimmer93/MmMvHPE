## 1. Panoptic Dataset Anchor Support

- [x] 1.1 Add optional `anchor_key` parameter parsing/validation in `datasets/panoptic_preprocessed_dataset_v1.py`.
- [x] 1.2 Implement anchor-coordinate conversion utilities for camera extrinsics and GT keypoints in `datasets/panoptic_preprocessed_dataset_v1.py`.
- [x] 1.3 Apply anchor conversion in sample assembly path with fail-fast checks for invalid/missing anchor camera.

## 2. Panoptic Comparison Config Suite

- [x] 2.1 Add PanopticHPE training config for temporal split + 1 RGB/1 LiDAR + foreground depth + non-piano allowlist.
- [x] 2.2 Add XFi training config for the same data regime, with `anchor_key: input_rgb` and Panoptic 19-joint output.
- [x] 2.3 Add four test configs (PanopticHPE/XFi × occluded/unoccluded allowlists).

## 3. Documentation and Validation

- [x] 3.1 Update Panoptic docs under `docs/` with anchor and comparison-config usage notes.
- [x] 3.2 Validate by loading the new configs and running dataset instantiation/smoke checks via `uv run`.
- [x] 3.3 Mark completed tasks in this file.
