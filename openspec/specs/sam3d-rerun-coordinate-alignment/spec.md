# sam3d-rerun-coordinate-alignment Specification

## Purpose
TBD - created by archiving change sam3d-gt-camera-coordinate. Update Purpose after archive.
## Requirements
### Requirement: Configurable GT Coordinate Space for SAM-3D-Body Rerun
The SAM-3D-Body rerun visualization SHALL support GT coordinate-space selection with two valid values: `canonical` and `camera`. The script SHALL apply exactly one GT coordinate space per run.

#### Scenario: Canonical GT mode is selected
- **WHEN** GT coordinate space is set to `canonical`
- **THEN** GT 3D keypoints SHALL be logged in canonical pelvis-centered coordinates without camera-frame transform

#### Scenario: Camera GT mode is selected
- **WHEN** GT coordinate space is set to `camera`
- **THEN** GT 3D keypoints SHALL be transformed per selected RGB view using that view's extrinsic matrix before logging

### Requirement: Camera-Space Transform Contract
For camera GT mode, the script SHALL transform each GT 3D point with `X_cam = R * X + t`, where `R` and `t` are extracted from the selected view extrinsic matrix with shape `(3, 4)`.

#### Scenario: Extrinsic matrix has valid shape
- **WHEN** a selected RGB view provides `extrinsic[:, :3]` and `extrinsic[:, 3]`
- **THEN** the script SHALL apply the transform using those values and preserve keypoint tensor shape `(..., J, 3)`

#### Scenario: Camera metadata is missing in camera mode
- **WHEN** GT coordinate space is `camera` and view camera extrinsics are absent for the sample
- **THEN** the script SHALL fail fast with an explicit error describing the missing `rgb_camera` metadata

### Requirement: GT Coordinate Space Metadata Logging
The rerun output SHALL record which GT coordinate space was used so downstream analysis can compare recordings unambiguously.

#### Scenario: Rerun metadata includes GT space
- **WHEN** the script writes `world/info/*` metadata
- **THEN** it SHALL include a field indicating `gt_coordinate_space` as `canonical` or `camera`

