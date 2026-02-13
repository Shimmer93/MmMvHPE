# sam3d-rerun-multiframe-visualization Specification

## Purpose
TBD - created by archiving change sam3d-rerun-multiframe. Update Purpose after archive.
## Requirements
### Requirement: Multi-Frame Visualization Control
The SAM-3D-Body rerun visualization script SHALL support a `--num-frames` CLI argument that controls how many frames from the selected sample window are visualized. The default value MUST preserve current behavior (`num_frames = 1`).

#### Scenario: Default behavior stays single-frame
- **WHEN** the script is run without `--num-frames`
- **THEN** it SHALL visualize exactly one frame using existing single-frame semantics

### Requirement: Deterministic Frame Selection
The script SHALL select frame indices deterministically from the sample temporal window and clip indices to valid bounds.

#### Scenario: Positive frame-index anchors contiguous range
- **WHEN** `--frame-index 3 --num-frames 4` is provided and the sample has at least 7 frames
- **THEN** the script SHALL visualize source frames `[3, 4, 5, 6]`

#### Scenario: Center anchor is used when frame-index is -1
- **WHEN** `--frame-index -1 --num-frames 5` is provided
- **THEN** the script SHALL use a center-anchored contiguous frame range clipped to valid window bounds

### Requirement: Timeline Logging Per Visualized Frame
The script SHALL log each visualized frame as a separate rerun timeline step while preserving standard namespace paths.

#### Scenario: Multiple frames generate multiple timeline steps
- **WHEN** `--num-frames 5` is used
- **THEN** rerun output SHALL contain five sequential timeline steps with frame data under `world/inputs/*`, `world/front/*`, `world/side/*`, and metadata under `world/info/*`

### Requirement: Render-Mode Compatibility in Multi-Frame Runs
Multi-frame visualization SHALL remain compatible with existing render modes (`overlay`, `auto`, `mesh`) without changing mode semantics.

#### Scenario: Overlay mode remains mesh-free across all frames
- **WHEN** `--render-mode overlay --num-frames 3` is used
- **THEN** the script SHALL skip mesh logging for all three timeline steps while still logging overlay outputs and per-frame metadata

