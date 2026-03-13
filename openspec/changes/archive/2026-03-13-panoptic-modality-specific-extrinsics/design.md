## Context

`PanopticPreprocessedDatasetV1` currently computes modality-specific extrinsics in `_camera_extrinsic(...)` using `M_color` for RGB and `M_depth` for depth/LiDAR. However, `_camera_params(...)` has an earlier precedence branch that uses `extrinsic_world_to_color` for all three modalities whenever that field is present in `meta/cameras_kinect_cropped.json`. On this machine, that field is present for all preprocessed Panoptic sequences, so depth and depth-derived LiDAR are effectively sharing RGB/world-to-color extrinsics at runtime.

This is a geometry contract issue in the dataset layer, not a config issue. Existing Panoptic configs do not toggle it. The value is injected during preprocessing/backfill by `tools/preprocess_panoptic_kinoptic.py` and `tools/patch_panoptic_preprocessed_extrinsics.py`, then consumed by `datasets/panoptic_preprocessed_dataset_v1.py`.

The immediate goal is to restore modality-correct sensor geometry for Panoptic runtime samples. This change does not attempt to solve RGB-mask-to-depth reprojection or silhouette differences between sensors. Correct extrinsics are a prerequisite for that later work, but not the same feature.

## Goals / Non-Goals

**Goals:**
- Make Panoptic runtime camera extrinsics modality-specific by default.
- Preserve current RGB behavior where `extrinsic_world_to_color` is available and valid.
- Make depth and depth-derived LiDAR use depth geometry instead of RGB geometry.
- Define explicit precedence and validation rules for `extrinsic_world_to_color`, `M_color`, and `M_depth`.
- Keep the Panoptic world frame anchored by the color-camera calibration, then derive world-to-depth from the color-depth relative transform.
- Validate the change with sensor-frame checks and visualization/debug outputs that can expose geometry drift.

**Non-Goals:**
- Reprojecting RGB masks into the depth image plane.
- Changing Panoptic preprocessing outputs on disk unless later required by a separate change.
- Generalizing the fix to other datasets.
- Adding config flags for legacy shared-extrinsic behavior unless implementation reveals a compatibility blocker.

## Decisions

### 1. Keep the fix inside `PanopticPreprocessedDatasetV1`
The change should be implemented in `datasets/panoptic_preprocessed_dataset_v1.py`, specifically in `_camera_params(...)` and its extrinsic resolution logic. No changes are needed in `main.py`, `datasets/data_api.py`, or model APIs.

Why:
- The incorrect behavior is localized to Panoptic sample assembly.
- Fixing it in the dataset keeps the geometry contract consistent for all downstream consumers.
- It avoids proliferating ad hoc corrections in visualization or model code.

Alternative considered:
- Patch visualization tools only. Rejected because training/evaluation would still use incorrect sensor geometry.

### 2. Use modality-specific precedence rules instead of one shared fallback
The loader should resolve extrinsics using explicit modality-aware precedence:
- RGB: use a valid Panoptic-world-to-color extrinsic from `extrinsic_world_to_color` or Panoptic calibration fallback.
- Depth/LiDAR: derive Panoptic-world-to-depth from the Panoptic-world-to-color extrinsic plus the Kinect local relative calibration:
  - `world_to_depth = inv(M_depth) @ M_color @ world_to_color`

Why:
- `extrinsic_world_to_color` is semantically color-specific.
- `M_color` and `M_depth` describe the relative modality-to-local geometry inside one Kinect node.
- The dataset runtime backprojection keeps points in the depth camera frame, so it needs a world-to-depth transform, not a local-frame transform.
- This matches the toolbox convention without redesigning the preprocessing format.

Alternative considered:
- Continue using `extrinsic_world_to_color` for depth because RGB/depth are synchronized and cropped together. Rejected because synchronization/cropping does not make the sensors share the same optical center.
- Compose depth world extrinsics from `M_world2sensor` and `M_depth`. Rejected because `M_world2sensor` is not the Panoptic world frame used by `extrinsic_world_to_color`.

### 3. Keep LiDAR aligned with depth geometry
For Panoptic, `input_lidar` is depth-derived, not an independent physical LiDAR sensor. Therefore its camera metadata should follow the depth extrinsic path, not the RGB path.

Why:
- The point cloud is generated from depth rays and `K_depth`.
- Using RGB extrinsics for a depth-derived point cloud mixes two different sensor models.

Alternative considered:
- Keep LiDAR on RGB/world-to-color because some visualizations were built around that frame. Rejected because it preserves the geometry bug instead of fixing it.

### 4. Fail fast if required color-world or depth-relative geometry is unavailable or malformed
If the dataset cannot build a valid Panoptic world-to-color extrinsic or a valid depth-relative transform from the camera metadata, it should raise an explicit error rather than silently falling back to color extrinsics for depth/LiDAR or to `M_world2sensor`.

Why:
- Silent fallback would hide incorrect geometry in the exact place we are trying to make strict.
- The project’s engineering rule is to fail fast on abnormal inputs.

Alternative considered:
- Add a permissive fallback to preserve legacy behavior. Rejected unless validation shows a real compatibility blocker.

### 5. Treat this as a runtime behavior change, not a metadata rewrite
The initial fix should change how existing preprocessed metadata is interpreted, not rewrite all `cameras_kinect_cropped.json` files.

Why:
- The metadata already contains `M_color`, `M_depth`, and a Panoptic-world-to-color extrinsic.
- A loader fix is cheaper, lower risk, and reversible.
- It avoids a mass backfill for all preprocessed sequences.

Alternative considered:
- Write new `extrinsic_world_to_depth` fields into every preprocessed sequence. Rejected for now because it is unnecessary to restore correct runtime behavior.

### 6. Validate with both numeric and visual checks
Validation should include:
- numeric inspection that RGB and depth camera extrinsics differ only by the small Kinect color-depth baseline for the same node after loading
- depth-derived point-cloud exports in the corrected sensor frame
- at least one Panoptic visualization/debug path that makes the coordinate-frame change observable

Why:
- This is a geometry fix; purely unit-test style validation is not enough.
- Visual outputs under `logs/` are already part of the project’s debugging workflow.

## Risks / Trade-offs

- [Existing Panoptic depth/LiDAR runs may change behavior] -> Mitigation: document the change as breaking for sensor-frame outputs and validate with before/after debug exports.
- [Some tools may have implicitly relied on shared RGB extrinsics] -> Mitigation: inspect the Panoptic visualization and evaluation scripts that consume `depth_camera` / `lidar_camera` and update docs where behavior changes are expected.
- [Using `M_world2sensor` would produce a different global frame than Panoptic world] -> Mitigation: derive depth extrinsics from Panoptic world-to-color plus `inv(M_depth) @ M_color` instead of from `M_world2sensor`.
- [Metadata inconsistency across sequences could surface once the loader stops falling back to color extrinsics] -> Mitigation: fail fast with explicit sequence/camera errors and validate across the preprocessed sequence set already on this machine.
- [Changing extrinsics without reprojection will not fully solve RGB/depth mask misalignment] -> Mitigation: document this clearly as a prerequisite fix, not the complete alignment solution.

## Migration Plan

- Update `datasets/panoptic_preprocessed_dataset_v1.py` to resolve modality-specific extrinsics.
- Add validation/debug utilities or extend existing ones to report loaded RGB vs depth extrinsics for a selected sequence/camera.
- Run checks on representative Panoptic sequences already present on this machine.
- Update `docs/` to record the corrected geometry contract and the fact that depth-derived LiDAR follows depth extrinsics.
- Rollback, if needed, is limited to reverting the loader behavior; no dataset files need to be rewritten.

## Open Questions

- None at the design level. The main remaining work is to express the modified dataset contract in the spec and then implement the loader change.
