## ADDED Requirements

### Requirement: Panoptic Kinoptic single-actor preprocessing entrypoint
The system SHALL provide a dedicated preprocessing script for Panoptic Kinoptic single-actor sequences that accepts explicit sequence selection and writes compact preprocessed outputs.

#### Scenario: Preprocess selected sequences
- **WHEN** the user runs the script with `--sequences` and/or `--sequence-list`
- **THEN** only the selected sequences are processed
- **AND** unselected sequences are not touched

### Requirement: Partial-download friendly execution
The script SHALL support preprocessing a subset of sequences during ongoing dataset downloads.

#### Scenario: Incomplete global dataset
- **WHEN** only a subset of sequence folders is complete
- **THEN** the user can run preprocessing for that subset without requiring all sequences

### Requirement: Sequence-level fail-fast validation
The script SHALL validate required sequence artifacts before processing and emit explicit errors for missing critical inputs.

#### Scenario: Missing required metadata
- **WHEN** a selected sequence lacks `ksynctables_<seq>.json` or `hdPose3d_stage1_coco19` annotations
- **THEN** the script reports a clear sequence-specific failure reason
- **AND** behavior follows `--continue-on-error` policy
