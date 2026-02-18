# HuMMan LiDAR Coordinate Contract

This project uses one LiDAR skeleton coordinate contract across HuMMan datasets:

- Reference contract: `datasets/humman_dataset_v2.py` + `PCCenterWithKeypoints`
- LiDAR point cloud input: `input_lidar` is centered by PC center
- LiDAR skeleton supervision: same centered frame as `input_lidar`

In practice, the following keys are aligned to the same frame:

- `gt_keypoints_lidar`
- `gt_keypoints_pc_centered_input_lidar`

## Dataset-specific behavior

- `HummanPreprocessedDatasetV2`
  - `PCCenterWithKeypoints` defines the centered LiDAR frame.
- `HummanPreprocessedDatasetV3`
  - JSON-loaded LiDAR skeletons are converted to the same centered frame.
  - `gt_keypoints_lidar` and `gt_keypoints_pc_centered_input_lidar` are consistent.
  - Legacy flags `align_lidar_skeleton_with_pc_center` and
    `lidar_skeleton_from_global_camera` are treated as deprecated compatibility
    options and do not change this contract.
- `HummanCameraDatasetV1`
  - LiDAR skeleton outputs are always PC-centered to match the same contract.
  - `output_pc_centered_lidar=false` is deprecated and ignored.

## Why this is required

Camera-head and fusion heads consume LiDAR skeletons assuming the same frame as
centered LiDAR point clouds. Mixing camera-frame/global-frame skeletons with
PC-centered LiDAR inputs causes unstable training and poor camera estimation.

## Quick sanity check

For samples with LiDAR:

- `gt_keypoints_lidar` should numerically match `gt_keypoints_pc_centered_input_lidar`
  (or differ only by tiny floating-point noise).
