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

### Requirement: HuMMan SAM3D visualization SHALL support official converted SMPL comparison outputs
HuMMan-focused SAM3D visualization in MMHPE SHALL support a comparison mode that includes official converted SMPL outputs alongside raw SAM/MHR outputs and GT SMPL24 outputs. When enabled, the visualization path SHALL use the official conversion wrapper rather than a heuristic joint-remapping adapter.

#### Scenario: HuMMan comparison visualization exports converted SMPL outputs
- **WHEN** a user runs a HuMMan SAM3D visualization/export command that requests SMPL comparison outputs
- **THEN** the command SHALL include official converted SMPL results in the generated output set together with raw SAM/MHR and GT SMPL24 views

#### Scenario: Missing official conversion support blocks HuMMan comparison visualization
- **WHEN** a user requests HuMMan converted-SMPL visualization but the required official conversion dependencies are unavailable
- **THEN** the visualization command SHALL fail explicitly instead of silently falling back to heuristic joint remapping
