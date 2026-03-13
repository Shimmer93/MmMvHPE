## 1. Transform Implementation

- [x] 1.1 Add one Panoptic-specific masking transform under `datasets/transforms/` that accepts `apply_to` with `rgb`, `depth`, or both.
- [x] 1.2 Implement deterministic mask path resolution from Panoptic sample metadata using sequence name, selected camera name, and synchronized frame identity.
- [x] 1.3 Implement RGB masking directly in the RGB plane and depth masking through RGB-to-depth reprojection with zero-fill semantics while preserving frame shapes and depth dtypes.
- [x] 1.4 Implement strict validation and fail-fast errors for missing masks, unreadable masks, unresolved metadata, and mask/frame shape mismatches.
- [x] 1.5 Export the new transform through `datasets/transforms/__init__.py` so it is available from config-driven pipelines.

## 2. Panoptic Pipeline Integration

- [x] 2.1 Verify the transform works against the current Panoptic sample layout produced by `datasets/panoptic_preprocessed_dataset_v1.py`, including single-frame and temporal-window samples.
- [x] 2.2 Add or update Panoptic config examples under `configs/` that show RGB-only masking, depth-only masking, and combined RGB+depth masking with the transform inserted before normalization/formatting.
- [x] 2.3 Confirm the transform does not require changes to `main.py`, `datasets/data_api.py`, or model APIs.

## 3. Validation

- [x] 3.1 Add targeted validation for successful masking on a real or fixture Panoptic sample for `apply_to=[rgb]`, `apply_to=[depth]`, and `apply_to=[rgb, depth]`.
- [x] 3.2 Add a validation export script under `scripts/` that writes sample RGB/mask alignment overlays plus masked and unmasked depth point clouds for visual inspection, including the reprojected depth-mask result.
- [x] 3.3 Generate a small set of real Panoptic validation artifacts from that script and record their output paths for manual review.
- [x] 3.4 Add targeted validation for failure cases: missing mask file, unreadable mask image, unresolved sequence/camera/frame metadata, and mask/frame shape mismatch.
- [x] 3.5 Run validation commands with `uv run` and record the exact commands/results needed to reproduce the checks.

## 4. Documentation

- [x] 4.1 Add documentation under `docs/` describing the transform purpose, expected Panoptic mask folder layout, runtime behavior, and failure modes.
- [x] 4.2 Document concrete YAML usage examples for train/val/test or visualization pipelines using the `apply_to` argument.
- [x] 4.3 Update the OpenSpec change artifacts only as needed to reflect implementation completion and validation outcomes.
