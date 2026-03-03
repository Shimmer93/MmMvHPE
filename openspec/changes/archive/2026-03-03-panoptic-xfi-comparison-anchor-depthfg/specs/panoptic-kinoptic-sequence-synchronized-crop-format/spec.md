## ADDED Requirements

### Requirement: Foreground depth compatibility for preprocessed Panoptic loading
Preprocessed Panoptic sequence layout SHALL support optional foreground-only depth loading without requiring sync-map regeneration.

#### Scenario: Foreground depth subdirectory remapping
- **WHEN** dataset loading enables foreground depth mode with a configured subdirectory name
- **THEN** depth frame resolution uses synchronized frame IDs from `meta/sync_map.json`
- **AND** depth files are resolved from `<sequence>/<foreground_depth_subdir>/kinect_X/<frame_id>.png`
- **AND** missing remapped files fail fast with explicit file-path errors

