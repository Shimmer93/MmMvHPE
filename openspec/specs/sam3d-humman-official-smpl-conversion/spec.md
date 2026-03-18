# Capability: sam3d-humman-official-smpl-conversion

## Purpose
Define the official MHR-to-SMPL conversion path used for HuMMan SAM3D evaluation and visualization in MMHPE.

## Requirements

### Requirement: HuMMan SAM3D official conversion SHALL fit SMPL from SAM/MHR outputs
The repository SHALL provide a HuMMan-targeted official conversion path that accepts SAM3D per-person output dictionaries and converts them to fitted SMPL outputs using the upstream MHR conversion workflow rather than a heuristic joint remap. The conversion path SHALL accept SAM3D outputs that include `pred_vertices` and `pred_cam_t`, and it SHALL expose fitted SMPL outputs in camera coordinates suitable for HuMMan SMPL24 evaluation and visualization.

#### Scenario: Convert one or more SAM3D outputs to SMPL
- **WHEN** a HuMMan SAM3D script passes one or more valid `SAM3DBodyEstimator.process_one_image(...)` output dictionaries into the official conversion wrapper
- **THEN** the wrapper SHALL produce fitted SMPL outputs for each person without requiring manual joint remapping logic in the caller
- **THEN** the wrapper SHALL expose converted SMPL joints and/or SMPL parameters with a consistent person-major batch dimension

#### Scenario: Required SAM3D output fields are missing
- **WHEN** SAM3D output dictionaries do not provide the fields required by the official converter
- **THEN** the wrapper SHALL fail fast with an error that identifies the missing fields

### Requirement: HuMMan SAM3D evaluation SHALL score converted SMPL24 outputs
HuMMan SAM3D evaluation in MMHPE SHALL compute HuMMan SMPL24 metrics from officially converted SMPL outputs instead of the heuristic MHR70-to-SMPL24 adapter. The conversion path SHALL preserve the current config-driven dataset selection and SHALL evaluate GT and prediction in the same camera coordinate convention used by the selected HuMMan RGB camera.

#### Scenario: HuMMan test sample is evaluated with official conversion
- **WHEN** HuMMan SAM3D evaluation runs on a config-selected test sample or batch of samples
- **THEN** prediction joints used for SMPL24 metrics SHALL come from the official MHR-to-SMPL conversion path
- **THEN** GT SMPL24 joints and converted prediction joints SHALL be compared in the same RGB camera coordinate frame

#### Scenario: Batched evaluation uses official conversion
- **WHEN** HuMMan SAM3D evaluation processes multiple frames in one run
- **THEN** the implementation SHALL support converting multiple SAM3D outputs in batches rather than requiring a one-frame conversion process for the entire evaluation job

### Requirement: HuMMan SAM3D comparison visualization SHALL include raw and converted outputs
The repository SHALL provide a HuMMan SAM3D comparison visualization/export path that shows GT SMPL24, raw SAM/MHR output, and official converted SMPL output for the same sample. The resulting artifacts SHALL be written under `logs/` and SHALL preserve enough metadata to identify the config, split, camera, and sample used.

#### Scenario: Export comparison artifacts for one HuMMan sample
- **WHEN** a user runs the HuMMan conversion visualization script for a valid sample
- **THEN** the script SHALL export at least one artifact showing GT SMPL24, raw SAM/MHR output, and converted SMPL output for the same sample
- **THEN** the artifact set SHALL be written to a deterministic directory under `logs/`

#### Scenario: Converted visualization uses the same official conversion path as evaluation
- **WHEN** HuMMan evaluation and HuMMan visualization are both run for the same SAM3D outputs
- **THEN** both paths SHALL use the same official conversion wrapper rather than separate conversion implementations
