## ADDED Requirements

### Requirement: H36M preprocessing entrypoint
The system SHALL add an H36M preprocessing routine to `tools/data_preprocess.py` that downsamples RGB frames to 480x640 and writes a compact SSD-friendly dataset.

#### Scenario: Run preprocessing
- **WHEN** the user invokes the H36M preprocessing mode with input root and output dir
- **THEN** the script writes resized RGB frames and metadata into the output directory

### Requirement: SSD output location
The preprocessing script SHALL support writing output to `/opt/data/h36m_preprocessed` by default or via a user-specified path.

#### Scenario: Default output root
- **WHEN** output path is omitted
- **THEN** the preprocessed dataset is stored under `/opt/data/h36m_preprocessed`
