## Purpose
Define optional anchor-coordinate behavior for preprocessed Panoptic samples so GT keypoints and camera extrinsics can be emitted in a selected sensor frame.

## Requirements
### Requirement: Optional anchor-coordinate output for Panoptic preprocessed samples
`PanopticPreprocessedDatasetV1` SHALL support an optional `anchor_key` mode that transforms ground-truth keypoints and camera extrinsics into the selected anchor sensor coordinate system.

#### Scenario: RGB-anchored output for RGB+LiDAR samples
- **WHEN** dataset config sets `anchor_key: input_rgb` and sample includes RGB and LiDAR cameras
- **THEN** `rgb_camera.extrinsic` is identity
- **AND** LiDAR/depth camera extrinsics are expressed relative to RGB camera coordinates
- **AND** `gt_keypoints` are returned in RGB camera coordinates

#### Scenario: Default behavior remains unchanged
- **WHEN** `anchor_key` is not enabled in dataset config
- **THEN** camera extrinsics and GT keypoints are returned in the dataset's existing coordinate convention
- **AND** existing Panoptic training configs continue to load without behavior regression

#### Scenario: Invalid anchor selection fails fast
- **WHEN** `anchor_key` is set to a modality not available in the configured sample modalities
- **THEN** dataset initialization or sample loading fails with explicit `ValueError`
