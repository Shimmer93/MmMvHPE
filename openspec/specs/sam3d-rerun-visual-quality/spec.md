# sam3d-rerun-visual-quality Specification

## Purpose
TBD - created by archiving change sam3d-rerun-visual-fixes. Update Purpose after archive.
## Requirements
### Requirement: RGB Image Integrity in SAM-3D-Body Rerun
The SAM-3D-Body rerun visualization SHALL preserve image content integrity from dataset sample to rerun logging. Logged RGB images MUST satisfy shape `(H, W, 3)` and maintain valid dynamic range without unintended quantization collapse (for example near-black images caused by incorrect scaling/casting).

#### Scenario: RGB frame logs with valid shape and dynamic range
- **WHEN** `scripts/visualize_sam3d_body_rerun.py` logs an RGB frame to `world/inputs/rgb/view_<i>/image`
- **THEN** the logged image SHALL use `(H, W, 3)` layout and retain non-degenerate value range consistent with the source frame format

### Requirement: Topology-Aware 2D Keypoint Overlay
The SAM-3D-Body rerun visualization SHALL render 2D keypoints with connected skeleton edges using the SAM-3D-Body keypoint definition/order, not point-only overlays.

#### Scenario: 2D overlay draws connected skeleton
- **WHEN** SAM-3D-Body returns `pred_keypoints_2d` for a detected person
- **THEN** rerun overlay output SHALL contain keypoint markers and line segments matching the configured SAM-3D-Body skeleton topology

### Requirement: Topology-Consistent 3D Skeleton Rendering
The SAM-3D-Body rerun visualization SHALL render 3D keypoints with connectivity based on the same SAM-3D-Body keypoint topology used for 2D overlays.

#### Scenario: 3D keypoints and edges follow SAM topology
- **WHEN** SAM-3D-Body returns `pred_keypoints_3d`
- **THEN** rerun 3D skeleton logs SHALL connect joint indices according to the SAM-3D-Body topology mapping and not an unrelated skeleton convention

### Requirement: Front/Side Transform Consistency
For each sample and render mode that includes 3D outputs, side-view mesh and side-view keypoints SHALL be derived from the same transformed front-view coordinates so they remain spatially aligned.

#### Scenario: Side-view keypoints align with side-view mesh
- **WHEN** the script logs prediction mesh and prediction keypoints to both `world/front/*` and `world/side/*`
- **THEN** the side-view keypoints SHALL align with the side-view mesh under the same transform pipeline

