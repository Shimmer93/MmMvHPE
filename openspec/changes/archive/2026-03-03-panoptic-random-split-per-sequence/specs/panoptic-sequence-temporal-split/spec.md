## ADDED Requirements

### Requirement: Temporal split mode partitions each sequence by synchronized frame order
When `split_to_use: temporal_split` is selected for Panoptic preprocessed data, the dataset SHALL split each selected sequence independently using synchronized common frame IDs sorted in ascending frame index order.

#### Scenario: Train partition uses sequence head frames
- **WHEN** `test_mode=False`, `ratio=0.8`, and a sequence has synchronized frame IDs `[f0, ..., f9]`
- **THEN** the dataset indexes frames `[f0, ..., f7]` for that sequence
- **AND** frames `[f8, f9]` are excluded from the train partition for that sequence

#### Scenario: Validation/test partition uses sequence tail frames
- **WHEN** `test_mode=True`, `ratio=0.8`, and a sequence has synchronized frame IDs `[f0, ..., f9]`
- **THEN** the dataset indexes frames `[f8, f9]` for that sequence
- **AND** frames `[f0, ..., f7]` are excluded from the validation/test partition for that sequence

### Requirement: Temporal split is deterministic and reproducible
Temporal partitioning SHALL be deterministic and SHALL NOT depend on random sequence permutation for Panoptic `temporal_split`.

#### Scenario: Repeated initialization yields identical membership
- **WHEN** the dataset is initialized multiple times with identical `data_root`, `split_config`, `split_to_use: temporal_split`, and `ratio`
- **THEN** each sequence contributes the same train/validation frame subsets in every run
- **AND** sample composition differences are allowed only from independent downstream sampling options (for example, `max_samples` random subsampling)

### Requirement: Sequence-level split remains available through explicit cross-subject sequence lists
Panoptic sequence-level partitioning SHALL be supported via explicit `sequences` lists under `cross_subject_split` train/validation entries in `configs/datasets/panoptic_split_config.yml`.

#### Scenario: Explicit sequence lists define train/validation sequence membership
- **WHEN** `split_to_use: cross_subject_split` and both train/validation entries define explicit `sequences`
- **THEN** only the listed sequences are selected for each partition
- **AND** no implicit random sequence assignment is applied by the dataset loader

### Requirement: Temporal split preserves multimodal synchronization assumptions
Temporal split boundaries SHALL be applied on synchronized common frame IDs so that indexed samples remain aligned across selected Panoptic modalities (`rgb`, `depth`, and depth-derived `lidar`).

#### Scenario: Multimodal sample remains frame-aligned after split
- **WHEN** a sample is indexed from `temporal_split` with both RGB and depth modalities enabled
- **THEN** RGB/depth paths for the sample map to the same body frame ID in the sequence sync metadata
- **AND** any derived lidar point cloud uses the same selected depth frame

### Requirement: Temporal split validation is fail-fast for invalid contracts
The dataset MUST fail fast for invalid temporal split configuration in strict mode.

#### Scenario: Invalid temporal ratio
- **WHEN** `split_to_use: temporal_split` and `ratio` is not within `(0, 1)`
- **THEN** dataset initialization raises an explicit error before indexing samples

#### Scenario: Empty partition for a selected sequence in strict mode
- **WHEN** `split_to_use: temporal_split`, `strict_validation=True`, and temporal cutoff yields zero frames for the requested partition in any selected sequence
- **THEN** dataset initialization raises an explicit error identifying the sequence and cutoff context
