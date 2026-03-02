## ADDED Requirements

### Requirement: SMPL pelvis-centered MPJPE metric
The system SHALL provide `SMPL_PCMPJPE` in `metrics/smpl_metrics.py` that computes mean per-joint position error after pelvis translation alignment and pelvis root-orientation alignment between prediction and GT for each sample.

#### Scenario: SMPL prediction differs only by root pose
- **WHEN** predicted SMPL joints differ from GT only by pelvis translation and pelvis root rotation
- **THEN** `SMPL_PCMPJPE` returns zero (or numerically near-zero) error within floating-point tolerance

#### Scenario: Missing required pelvis orientation input
- **WHEN** required pelvis/root orientation inputs are absent or have invalid shape
- **THEN** the metric computation fails fast with an explicit error describing missing/invalid assumptions

### Requirement: Direct-keypoint pelvis-centered MPJPE metric
The system SHALL provide `PCMPJPE` in `metrics/mpjpe.py` that computes MPJPE after pelvis translation alignment and pelvis orientation alignment, where non-SMPL orientation is derived using the same convention as `remove_root_rotation` in `datasets/panoptic_preprocessed_dataset_v1.py`.

#### Scenario: Keypoint prediction differs only by pelvis translation and derived pelvis orientation
- **WHEN** predicted keypoints differ from GT only by pelvis translation and root orientation under the shared convention
- **THEN** `PCMPJPE` returns zero (or numerically near-zero) error within floating-point tolerance

#### Scenario: Required keypoints for orientation are unavailable
- **WHEN** keypoints required by the orientation convention are missing or invalid
- **THEN** `PCMPJPE` fails fast with an explicit error instead of silently falling back to another alignment

### Requirement: Config-driven metric enablement
The system SHALL allow `PCMPJPE` and `SMPL_PCMPJPE` to be declared in experiment config metric lists without changing existing metric names or evaluation wiring.

#### Scenario: Existing MPJPE metrics remain available
- **WHEN** a config includes existing metrics plus the new pelvis-centered metrics
- **THEN** evaluation computes and logs both existing and new metrics in the same run outputs under `logs/`
