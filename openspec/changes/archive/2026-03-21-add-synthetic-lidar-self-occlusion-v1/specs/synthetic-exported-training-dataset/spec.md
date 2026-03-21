## MODIFIED Requirements

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
