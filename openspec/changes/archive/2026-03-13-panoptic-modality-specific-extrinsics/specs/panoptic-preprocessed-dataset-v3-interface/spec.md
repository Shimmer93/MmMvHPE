## MODIFIED Requirements

### Requirement: Calibration metadata availability in samples
The dataset class SHALL load and expose preprocessed calibration metadata required by downstream projection/transformation logic. RGB samples SHALL expose color-camera intrinsics and color-camera extrinsics. Depth samples and depth-derived LiDAR samples SHALL expose depth-camera intrinsics and depth-camera extrinsics. The dataset class SHALL derive world-to-depth extrinsics from the Panoptic world-to-color extrinsic and the Kinect relative calibration, and SHALL NOT reuse `extrinsic_world_to_color` directly as the runtime extrinsic for depth or depth-derived LiDAR.

#### Scenario: Access RGB camera metadata from a sample
- **WHEN** a training or evaluation sample includes RGB modality data
- **THEN** the sample SHALL expose RGB camera metadata using color intrinsics and color extrinsics
- **AND** camera naming SHALL remain normalized to the style expected by existing transform and model paths

#### Scenario: Access depth camera metadata from a sample
- **WHEN** a training or evaluation sample includes depth modality data
- **THEN** the sample SHALL expose depth camera metadata using depth intrinsics and depth extrinsics
- **AND** the runtime extrinsic SHALL be derived as `inv(M_depth) @ M_color @ world_to_color`
- **AND** the runtime extrinsic SHALL differ from RGB/world-to-color extrinsic only by the Kinect color-depth relative calibration encoded by `M_color` and `M_depth`

#### Scenario: Access depth-derived LiDAR camera metadata from a sample
- **WHEN** a training or evaluation sample includes depth-derived LiDAR data produced from Panoptic depth frames
- **THEN** the sample SHALL expose LiDAR camera metadata using the same extrinsic geometry as the source depth modality
- **AND** the runtime extrinsic SHALL NOT be taken from RGB/world-to-color metadata by default

#### Scenario: Missing or malformed depth extrinsic metadata
- **WHEN** a Panoptic sample requires depth or depth-derived LiDAR camera metadata and the preprocessed camera metadata cannot provide a valid world-to-color extrinsic or a valid `M_color` / `M_depth` relative transform
- **THEN** dataset loading SHALL fail with an explicit error identifying the affected sequence and camera
- **AND** it SHALL NOT silently fall back to RGB/world-to-color extrinsics for that depth-based modality
- **AND** it SHALL NOT silently use `M_world2sensor` as a substitute for the Panoptic world frame
