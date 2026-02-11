# Capability: fusionformer-modality-baseline

## Purpose
Define the FusionFormer-style baseline that fuses modality tokens over time and outputs 3D keypoints only.

## Requirements

### Requirement: FusionFormer baseline outputs 3D keypoints
The system SHALL provide a FusionFormer-style model that fuses multi-view modality tokens over time and outputs 3D keypoints only.

#### Scenario: Training forward pass
- **WHEN** the model receives RGB 2D pose tokens and PC 3D pose tokens for a sample sequence
- **THEN** it returns a 3D keypoint prediction tensor with the same joint order as Humman (24 joints)

### Requirement: Camera-parameter-free fusion
The FusionFormer baseline SHALL ignore camera intrinsics and extrinsics during fusion and regression.

#### Scenario: Input includes camera parameters
- **WHEN** the data sample contains camera intrinsics/extrinsics
- **THEN** the baseline model does not consume them for fusion or prediction

### Requirement: Modality-as-view fusion
The baseline SHALL treat RGB and depth/point cloud streams as distinct “views” and fuse them in a unified Transformer block.

#### Scenario: Two modality inputs
- **WHEN** RGB 2D tokens and PC 3D tokens are present
- **THEN** both modalities are fused jointly across time in the encoder before regression
