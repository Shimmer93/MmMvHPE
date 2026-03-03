# panoptic-model-comparison-configs Specification

## Purpose

Define reproducible Panoptic comparison config contracts for `PanopticHPE` and `XFi` under matched RGB+LiDAR data settings, including occluded and unoccluded test subsets.

## Requirements

### Requirement: Panoptic comparison training configs for PanopticHPE and XFi
The repository SHALL provide Panoptic training configs for `PanopticHPE` and `XFi` that use the same Panoptic preprocessed root, temporal split, foreground depth-derived LiDAR input, and one RGB plus one LiDAR camera per sample.

#### Scenario: Comparable training data regime
- **WHEN** training configs are loaded for the comparison experiments
- **THEN** both configs use `PanopticPreprocessedDatasetV1` with `split_to_use: temporal_split`
- **AND** both configs enable `use_foreground_depth`
- **AND** both configs exclude the three piano sequences from sequence selection

#### Scenario: XFi RGB-anchor requirement
- **WHEN** the XFi Panoptic training config is loaded
- **THEN** dataset params set `anchor_key: input_rgb`
- **AND** GT keypoints are provided in RGB camera coordinates for training/evaluation

### Requirement: Occluded vs unoccluded Panoptic test config set
The repository SHALL provide four Panoptic test configs: PanopticHPE-occluded, PanopticHPE-unoccluded, XFi-occluded, and XFi-unoccluded.

#### Scenario: Occluded test subset
- **WHEN** an occluded test config is loaded
- **THEN** test sequence allowlist is exactly `{170915_office1, 170407_office2, 171026_cello3}`

#### Scenario: Unoccluded test subset
- **WHEN** an unoccluded test config is loaded
- **THEN** test sequence allowlist contains all non-piano and non-occluded comparison sequences
