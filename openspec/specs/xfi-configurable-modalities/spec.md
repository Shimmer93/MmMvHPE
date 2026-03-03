# xfi-configurable-modalities Specification

## Purpose
TBD - created by archiving change configurable-xfi-modalities. Update Purpose after archive.
## Requirements
### Requirement: XFi Aggregator Modality Configuration
The system SHALL allow XFi aggregation to declare active modalities explicitly in config via `aggregator.params.active_modalities` as an ordered list containing only `rgb`, `depth`, `mmwave`, and `lidar`.

#### Scenario: Valid explicit modality list
- **WHEN** an XFi config sets `aggregator.params.active_modalities: ['rgb', 'lidar']`
- **THEN** the model SHALL build and run aggregation using only RGB and LiDAR branches in canonical modality mapping

#### Scenario: Invalid modality name
- **WHEN** an XFi config includes an unsupported modality token in `active_modalities`
- **THEN** initialization SHALL fail with an explicit error that lists supported modality names

### Requirement: Canonical Modality Ordering Contract
The system SHALL use a single canonical modality order `['rgb', 'depth', 'mmwave', 'lidar']` for mapping configured modalities, extracted features, and XFi projector branches.

#### Scenario: Mixed present and absent modalities
- **WHEN** only RGB and LiDAR are configured as active modalities
- **THEN** the projector SHALL map RGB features to RGB branch and LiDAR features to LiDAR branch without attempting depth or mmWave branch computation

### Requirement: Fail-Fast Modality Consistency Validation
The system SHALL validate consistency among configured active modalities and runtime aggregator feature inputs before branch projection proceeds.

#### Scenario: Missing feature tensor for required modality
- **WHEN** `active_modalities` includes `lidar` but LiDAR feature input is `None` at aggregator forward
- **THEN** forward execution SHALL fail with an explicit runtime error identifying the missing modality

#### Scenario: Invalid feature shape for required modality
- **WHEN** `active_modalities` includes `rgb` but RGB feature tensor shape is incompatible with XFi projector expectations
- **THEN** forward execution SHALL fail with an explicit runtime error describing expected and received shapes

### Requirement: Backward Compatibility for Existing XFi Configs
The system SHALL support existing XFi configs that omit `active_modalities` by deriving active modalities from non-`None` aggregator feature inputs in canonical order.

#### Scenario: Legacy RGB-depth config without active_modalities
- **WHEN** an existing config does not define `active_modalities` and aggregator receives non-`None` RGB and depth features
- **THEN** XFi aggregation SHALL behave as RGB+depth and emit a one-time warning recommending explicit `active_modalities`

