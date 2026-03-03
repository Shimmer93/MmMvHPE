## ADDED Requirements

### Requirement: Panoptic preprocessed dataset class with V3-compatible interface
The system SHALL provide a Panoptic preprocessed dataset class that consumes sequence-preserving preprocessed Kinoptic single-actor data and exposes the same training-facing sample interface contract used by `humman_dataset_v3`.

#### Scenario: Load a sample through the dataset API
- **WHEN** a config selects the Panoptic preprocessed dataset type through `datasets/data_api.py`
- **THEN** dataset construction succeeds without requiring model/loss/metric API changes
- **AND** each returned sample follows the `humman_dataset_v3`-compatible field contract

### Requirement: Default Panoptic preprocessed root
The dataset class SHALL default its data root to `/opt/data/panoptic_kinoptic_single_actor_cropped` unless explicitly overridden in config.

#### Scenario: Instantiate without custom root
- **WHEN** the dataset is created without `data_root` override
- **THEN** it resolves the default root `/opt/data/panoptic_kinoptic_single_actor_cropped`

### Requirement: Hard-fail validation for required sequence artifacts
The dataset class MUST validate required per-sequence artifacts at index build time and MUST fail with explicit errors by default when artifacts are missing or malformed.

#### Scenario: Missing required files in selected sequence
- **WHEN** a selected sequence is missing required artifacts (for example synchronized metadata, cropped calibration metadata, RGB/depth frame lists, or GT keypoints)
- **THEN** dataset initialization raises an explicit error identifying the sequence and missing artifact
- **AND** initialization does not silently skip the invalid sequence under default behavior

### Requirement: Calibration metadata availability in samples
The dataset class SHALL load and expose preprocessed calibration metadata required by downstream projection/transformation logic.

#### Scenario: Access camera metadata from a sample
- **WHEN** a training/eval sample is returned
- **THEN** camera intrinsics/extrinsics metadata from preprocessed calibration files is present in the sample metadata contract
- **AND** camera naming is normalized to the style expected by existing transform/model paths
