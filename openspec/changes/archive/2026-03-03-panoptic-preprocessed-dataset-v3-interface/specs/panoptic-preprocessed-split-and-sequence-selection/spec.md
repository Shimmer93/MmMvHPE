## ADDED Requirements

### Requirement: Config-file-driven split selection
The dataset SHALL support split selection from a split config file in the same workflow style as `configs/datasets/humman_split_config.yml`.

#### Scenario: Use explicit split lists from config
- **WHEN** `split_config` is provided and contains explicit sequence lists for the requested split
- **THEN** only sequences listed for that split are indexed
- **AND** sequences not in the requested split are excluded

### Requirement: Deterministic sequence-level ratio split fallback
When explicit split sequence lists are absent, the dataset SHALL derive train/val/test splits using deterministic sequence-level ratio splitting.

#### Scenario: Build fallback split from ratios
- **WHEN** the split config provides ratios but not explicit sequence lists
- **THEN** the dataset sorts available sequences in a stable order and applies ratio-based assignment at sequence level
- **AND** repeated runs with the same inputs and seed produce identical split membership

### Requirement: User-provided sequence subset filtering
The dataset SHALL allow users to restrict preprocessing/loading to a specified sequence subset and keep sequence boundaries intact.

#### Scenario: Apply sequence allowlist
- **WHEN** a user provides a sequence list/filter in config
- **THEN** dataset indexing is restricted to the intersection of split-selected and user-selected sequences
- **AND** no frames from non-selected sequences are loaded

### Requirement: Hard-fail default for split/selection validation
The dataset MUST hard-fail by default when split/selection references unknown sequences or when selected sequences fail required artifact validation.

#### Scenario: Unknown sequence in split config
- **WHEN** the requested split includes a sequence that does not exist under the configured Panoptic preprocessed root
- **THEN** dataset initialization raises an explicit error with the missing sequence name
- **AND** initialization does not continue silently by default
