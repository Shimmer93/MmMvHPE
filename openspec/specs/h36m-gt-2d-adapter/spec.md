# Capability: h36m-gt-2d-adapter

## Purpose
Derive 2D keypoints for H36M multiview RGB using perspective projection.

## Requirements

### Requirement: GT 2D keypoints via perspective projection
The system SHALL derive per-view 2D keypoints from GT 3D camera-space joints using perspective projection with camera intrinsics.

#### Scenario: Perspective projection
- **WHEN** GT 3D joints and camera intrinsics are available for a view
- **THEN** the adapter outputs 2D keypoints computed as (x = fx * X/Z + cx, y = fy * Y/Z + cy)

### Requirement: Per-frame 2D keypoint sequences
The adapter SHALL output per-frame 2D keypoint sequences aligned to the RGB sequence length.

#### Scenario: Sequence alignment
- **WHEN** the RGB sequence length is T
- **THEN** the adapter outputs 2D keypoints for T frames with the same joint order
