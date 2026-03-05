## MODIFIED Requirements

### Requirement: SMPL pelvis-centered MPJPE metric
The system SHALL provide `SMPL_PCMPJPE` in `metrics/smpl_metrics.py` that computes mean per-joint position error after pelvis translation alignment between prediction and GT for each sample.

#### Scenario: SMPL prediction differs only by pelvis translation
- **WHEN** predicted SMPL joints differ from GT only by pelvis translation
- **THEN** `SMPL_PCMPJPE` returns zero (or numerically near-zero) error within floating-point tolerance

#### Scenario: SMPL prediction differs by root rotation
- **WHEN** predicted SMPL joints differ from GT by root rotation after pelvis translation centering
- **THEN** `SMPL_PCMPJPE` reports non-zero error reflecting the remaining rotation mismatch

### Requirement: Direct-keypoint pelvis-centered MPJPE metric
The system SHALL provide `PCMPJPE` in `metrics/mpjpe.py` that computes MPJPE after pelvis translation alignment only.

#### Scenario: Keypoint prediction differs only by pelvis translation
- **WHEN** predicted keypoints differ from GT only by pelvis translation
- **THEN** `PCMPJPE` returns zero (or numerically near-zero) error within floating-point tolerance

#### Scenario: Keypoint prediction differs by global rotation
- **WHEN** predicted keypoints differ from GT by global rotation after pelvis translation centering
- **THEN** `PCMPJPE` reports non-zero error reflecting the rotation mismatch

### Requirement: Config-driven metric enablement
The system SHALL allow `PCMPJPE` and `SMPL_PCMPJPE` to be declared in experiment config metric lists without changing existing metric names or evaluation wiring.

#### Scenario: Existing MPJPE metrics remain available
- **WHEN** a config includes existing metrics plus the pelvis-centered metrics
- **THEN** evaluation computes and logs both existing and pelvis-centered metrics in the same run outputs under `logs/`
