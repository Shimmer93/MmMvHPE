# synthetic-exported-training-dataset Specification

## Purpose
Define the training-facing contract for loading sample-centric synthetic exports directly into the existing MMHPE training pipelines.

## Requirements
### Requirement: Synthetic Exported Dataset Shall Load Minimal Sample-Centric Exports
The system SHALL provide a dataset class under `datasets/` that reads synthetic samples directly from the sample-centric export roots produced by the synthetic-data pipeline, without requiring the synthetic data to be reshaped into HuMMan-style or Panoptic-style dataset trees.

When multiple LiDAR artifact versions coexist within one synthetic sample, the dataset SHALL allow the selected LiDAR version to be chosen from dataset or config parameters and SHALL resolve `input_lidar` from that requested version explicitly.

#### Scenario: Enumerate accepted synthetic samples
- **WHEN** a dataset is created with a synthetic root containing per-sample `manifest.json` and `exports/export_manifest.json` files
- **THEN** it SHALL index only accepted sample directories with the requested target format available

#### Scenario: Fail on missing required sample artifacts
- **WHEN** a required manifest, camera JSON, NPY artifact, or source RGB path is missing for an indexed sample
- **THEN** the dataset SHALL raise an explicit error instead of silently fabricating fallback data

#### Scenario: Load the requested LiDAR artifact version
- **WHEN** a dataset is constructed with a LiDAR-version selection parameter and the selected synthetic sample contains both `v0-a` and `v1` LiDAR artifacts
- **THEN** the dataset SHALL load `input_lidar` from the requested version instead of implicitly choosing one

#### Scenario: Fail on unavailable LiDAR artifact version
- **WHEN** a dataset requests a LiDAR artifact version that is not present in an otherwise valid synthetic sample
- **THEN** the dataset SHALL raise an explicit error identifying the missing LiDAR version

### Requirement: Synthetic Exported Dataset Shall Emit Real-Pipeline-Compatible Sample Keys
The dataset SHALL emit the same training-facing sample keys required by the existing MMHPE stage-1 and stage-2 pipelines, including RGB, LiDAR, camera dictionaries, GT keypoints, and stage-2 camera-head supervision keys.

#### Scenario: HuMMan-style sample contract
- **WHEN** the dataset is created with `target_format: humman`
- **THEN** each sample SHALL expose `input_rgb`, `input_lidar`, `rgb_camera`, `lidar_camera`, `gt_keypoints`, `gt_smpl_params`, `gt_global_orient`, `gt_pelvis`, `gt_keypoints_2d_rgb`, `gt_keypoints_lidar`, and `gt_keypoints_pc_centered_input_lidar`

#### Scenario: Panoptic-style sample contract
- **WHEN** the dataset is created with `target_format: panoptic`
- **THEN** each sample SHALL expose `input_rgb`, `input_lidar`, `rgb_camera`, `lidar_camera`, `gt_keypoints`, `gt_smpl_params`, `gt_global_orient`, `gt_pelvis`, `gt_keypoints_2d_rgb`, `gt_keypoints_lidar`, and `gt_keypoints_pc_centered_input_lidar`, with `gt_keypoints` using Panoptic joints19 topology

### Requirement: Synthetic Exported Dataset Shall Derive Panoptic LiDAR-Side Supervision Without Replicating Export Files
For Panoptic-style synthetic training, the dataset SHALL derive LiDAR-side camera and 3D supervision from the base synthetic manifest and artifacts instead of requiring extra replicated files inside the Panoptic export directory.

#### Scenario: Derive Panoptic LiDAR camera and LiDAR-frame joints
- **WHEN** a Panoptic-style sample is loaded
- **THEN** the dataset SHALL reconstruct `lidar_camera`, `gt_camera_lidar`, `gt_keypoints_lidar`, and `gt_keypoints_pc_centered_input_lidar` from the base synthetic sample artifacts and exported Panoptic world/new-world joints

#### Scenario: Preserve minimal on-disk exports
- **WHEN** the dataset loads a Panoptic-style sample
- **THEN** it SHALL not require the synthetic exporter to have created a second replicated LiDAR-camera payload tree under `exports/panoptic`

### Requirement: Synthetic Exported Dataset Shall Be Restricted To Single-Frame Synthetic Samples
The initial synthetic training dataset SHALL support the current exported synthetic contract only for `seq_len=1`.

#### Scenario: Reject unsupported temporal settings
- **WHEN** the dataset is constructed with `seq_len` other than `1`
- **THEN** it SHALL raise an explicit error describing that only single-frame synthetic exports are supported
