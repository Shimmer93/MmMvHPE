# Capability: h36m-multiview-rgb

## Purpose
Support multiview RGB sampling for H36M with synchronized per-camera sequences.

## Requirements

### Requirement: Multi-camera RGB sequences per sample
The system SHALL provide an H36M dataset path that returns synchronized RGB sequences for multiple cameras per sample.

#### Scenario: Multi-camera selection
- **WHEN** the config sets cameras to ['01','02','03','04']
- **THEN** each sample includes RGB frames for all selected cameras in a consistent view order

### Requirement: View-stacked RGB tensor shape
The dataset SHALL return RGB frames in a view-stacked format suitable for multiview fusion.

#### Scenario: View-stacked output
- **WHEN** a sample is loaded with V cameras and T frames
- **THEN** the RGB output is shaped as [V, T, H, W, 3]
