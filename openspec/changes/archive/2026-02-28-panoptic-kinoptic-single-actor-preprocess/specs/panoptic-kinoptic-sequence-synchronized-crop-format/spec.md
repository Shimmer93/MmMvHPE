## ADDED Requirements

### Requirement: Sequence-preserving output layout
Preprocessed Panoptic Kinoptic outputs SHALL remain grouped by sequence and SHALL NOT mix frames from different sequences.

#### Scenario: Output structure check
- **WHEN** preprocessing completes for sequence `S`
- **THEN** all produced artifacts for `S` are located under `<out_root>/S/`
- **AND** no artifact for `S` is written under another sequence folder

### Requirement: Synchronization by time metadata
The preprocessing pipeline SHALL synchronize RGB/depth/body frames using sequence timing metadata (`univTime` + sync tables), not raw index matching.

#### Scenario: Build synchronized sample set
- **WHEN** the script processes a sequence with valid sync tables and body annotations
- **THEN** synchronized frame pairs/triples are derived by nearest-time matching within configured tolerance
- **AND** unmatched frames outside tolerance are dropped and reported

### Requirement: HuMMan-style cropped RGB output
The preprocessing pipeline SHALL produce square-cropped RGB outputs resized to configured dimensions, aligned with the project's HuMMan-cropped preprocessing style.

#### Scenario: Cropped RGB sample
- **WHEN** inspecting a processed RGB output frame
- **THEN** the frame has configured output resolution
- **AND** crop metadata is recorded for reproducibility

### Requirement: Synchronization manifest and preprocessing metadata
Each processed sequence SHALL include metadata files describing synchronized frame mapping and crop parameters.

#### Scenario: Metadata presence
- **WHEN** a sequence finishes preprocessing
- **THEN** sequence-local metadata files include synchronized frame mapping and crop parameter records
