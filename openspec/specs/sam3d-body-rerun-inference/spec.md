# sam3d-body-rerun-inference Specification

## Purpose
TBD - created by archiving change rerun-sam3d-body-visualization. Update Purpose after archive.
## Requirements
### Requirement: SAM-3D-Body Rerun Inference Entry Script
The repository SHALL provide a SAM-3D-Body rerun visualization entry script under `scripts/` that loads dataset configuration from a config file, selects samples by split/index, runs SAM-3D-Body inference, and logs results with the shared rerun pipeline.

#### Scenario: Run SAM-3D-Body inference from config-driven sample selection
- **WHEN** the user runs the SAM-3D-Body rerun script with a valid config path, split name, and sample index
- **THEN** the script SHALL load the specified dataset sample, execute inference, and write rerun output without requiring manual code edits

### Requirement: Explicit Render Mode Control
The SAM-3D-Body rerun script SHALL expose an explicit CLI rendering mode switch so behavior is deterministic across machines. It SHALL support at least `mesh`, `overlay`, and `auto` modes.

#### Scenario: Forced overlay mode bypasses mesh rendering path
- **WHEN** the script is launched with `--render-mode overlay`
- **THEN** the script SHALL skip mesh rendering and only log overlay-compatible outputs while still logging inference metadata

### Requirement: Checkpoint and Asset Contract for SAM-3D-Body
The SAM-3D-Body rerun script SHALL load required model assets from `/opt/data/SAM_3dbody_checkpoints/` using `model_config.yaml`, `model.ckpt`, and `mhr_model.pt`.

#### Scenario: Missing checkpoint asset fails with actionable error
- **WHEN** any required checkpoint file is missing from `/opt/data/SAM_3dbody_checkpoints/`
- **THEN** the script SHALL fail fast with an error message that names the missing file and required root directory

