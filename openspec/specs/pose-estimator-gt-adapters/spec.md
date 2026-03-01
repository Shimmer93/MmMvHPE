# Capability: pose-estimator-gt-adapters

## Purpose
Provide GT pose adapters that supply per-frame 2D/3D keypoints as stand-ins for external estimators.

## Requirements

### Requirement: GT 2D pose adapter for RGB
The system SHALL provide a GT 2D pose adapter that supplies per-frame 2D keypoints for RGB inputs as a stand-in for an external 2D pose estimator.

#### Scenario: RGB GT 2D available
- **WHEN** a data sample includes RGB frames and GT 2D keypoints
- **THEN** the adapter outputs 2D keypoints aligned to the sample’s joint order

### Requirement: GT 3D pose adapter for point cloud
The system SHALL provide a GT 3D pose adapter that supplies per-frame 3D keypoints for point cloud inputs as a stand-in for an external 3D pose estimator.

#### Scenario: PC GT 3D available
- **WHEN** a data sample includes point cloud frames and GT 3D keypoints
- **THEN** the adapter outputs 3D keypoints aligned to the sample’s joint order

### Requirement: Adapter outputs are configurable
The system SHALL allow configuration to enable/disable GT adapters per modality.

#### Scenario: Disabling GT adapters
- **WHEN** GT adapters are disabled in config
- **THEN** the baseline does not attach GT pose outputs and expects estimator outputs from elsewhere
