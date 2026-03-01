# Capability: mamba4d-pointcloud-pose-estimator

## Purpose
Define a HuMMan point-cloud-only sequence pose estimator using Mamba4D that predicts per-frame 3D keypoints.

## Requirements

### Requirement: Mamba4D sequence pose estimator
The system SHALL use the existing Mamba4D point-cloud encoder to produce per-frame features and regress a sequence of 3D keypoints shaped `B, T, J, 3` for HuMMan.

#### Scenario: Sequence regression from point clouds
- **WHEN** a point-cloud sequence is passed through the estimator
- **THEN** the output SHALL be a sequence of 3D keypoints with length T

### Requirement: Sequence MSE loss
The training pipeline SHALL compute uniform per-frame MSE loss between predicted and GT 3D keypoint sequences.

#### Scenario: Uniform loss over time
- **WHEN** computing the keypoint loss
- **THEN** each timestep SHALL contribute equally to the MSE

### Requirement: Configurable HuMMan training
The system SHALL provide a HuMMan config that trains the Mamba4D sequence pose estimator using point-cloud inputs only.

#### Scenario: Point-cloud-only training
- **WHEN** the HuMMan config selects the Mamba4D estimator
- **THEN** training SHALL run using only point-cloud inputs and sequence loss
