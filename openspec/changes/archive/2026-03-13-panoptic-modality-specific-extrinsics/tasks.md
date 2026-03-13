## 1. Dataset Loader Fix

- [x] 1.1 Update `datasets/panoptic_preprocessed_dataset_v1.py` so `_camera_params(...)` resolves RGB extrinsics from Panoptic world-to-color geometry and depth/LiDAR extrinsics from `inv(M_depth) @ M_color @ world_to_color`.
- [x] 1.2 Remove the current depth/LiDAR runtime fallback to `extrinsic_world_to_color` when valid depth geometry is available.
- [x] 1.3 Add strict validation and explicit errors for missing or malformed depth extrinsic metadata needed by depth or depth-derived LiDAR samples.

## 2. Validation and Debugging

- [x] 2.1 Add a small validation/debug utility under `scripts/` or `tools/` that reports loaded RGB vs depth extrinsics for a selected Panoptic sequence/camera.
- [x] 2.2 Validate on representative preprocessed Panoptic sequences from this machine that RGB and depth extrinsics now differ only by the expected Kinect color-depth relative baseline where `M_color` and `M_depth` differ.
- [x] 2.3 Export at least one corrected depth-derived point-cloud or sensor-frame debug sample under `logs/` so the geometry change is visually inspectable.
- [x] 2.4 Run reproducible `uv run` validation commands and record the exact commands/results.

## 3. Compatibility Review

- [x] 3.1 Check Panoptic visualization or evaluation utilities that consume `depth_camera` or `lidar_camera` metadata for assumptions about shared RGB extrinsics.
- [x] 3.2 Confirm no changes are required in `main.py`, `datasets/data_api.py`, model APIs, or YAML config keys for the loader fix.

## 4. Documentation

- [x] 4.1 Add documentation under `docs/` describing the corrected Panoptic modality-specific extrinsic contract and the breaking impact on depth/LiDAR sensor-frame outputs.
- [x] 4.2 Document that RGB-mask-to-depth reprojection is explicitly out of scope for this change.
- [x] 4.3 Update the OpenSpec change artifacts only as needed to reflect implementation completion and validation outcomes.
