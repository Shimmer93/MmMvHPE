# Capability: h36m-fusionformer-config

## Purpose
Define the H36M FusionFormer experiment configuration aligned with paper settings.

## Requirements

### Requirement: H36M FusionFormer config (paper-aligned)
The system SHALL provide an H36M FusionFormer config with T=27, B=2, 17 joints, and all cameras enabled by default.

#### Scenario: Default settings
- **WHEN** the config is loaded without overrides
- **THEN** it uses seq_len=27, num_blocks=2, num_joints=17, and cameras ['01','02','03','04']

### Requirement: Configurable cameras and sequence length
The config SHALL allow users to override cameras and sequence length for debugging or ablation.

#### Scenario: Override cameras
- **WHEN** the config sets cameras to a subset
- **THEN** the dataset uses only the selected cameras for multiview loading
