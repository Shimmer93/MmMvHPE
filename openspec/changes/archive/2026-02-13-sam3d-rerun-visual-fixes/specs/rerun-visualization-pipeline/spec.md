## MODIFIED Requirements

### Requirement: Config-Driven Input Modality/View Layout
The visualization pipeline SHALL build input panels and rerun entity paths from dataset configuration values instead of fixed modality/view assumptions. It SHALL read `modality_names` and per-modality camera definitions from config, supporting both explicit count keys (for example `rgb_cameras_per_sample`) and camera lists (for example `rgb_cameras`) with validation for consistency.

#### Scenario: Camera list drives modality view count
- **WHEN** a dataset config defines `modality_names: ["rgb"]` and `rgb_cameras: ["cam0", "cam1", "cam2"]`
- **THEN** rerun input entities SHALL be created for `world/inputs/rgb/view_0..2`

#### Scenario: Inconsistent camera configuration fails fast
- **WHEN** a dataset config defines both `rgb_cameras_per_sample` and `rgb_cameras` with different counts
- **THEN** layout construction SHALL fail with an explicit error that names the inconsistent keys

### Requirement: Standardized Rerun Logging Namespaces
The visualization pipeline SHALL log inputs, outputs, and metadata in standardized namespaces shared across scripts to support side-by-side comparison and tooling.

#### Scenario: Metadata and render-mode info are logged consistently
- **WHEN** a visualization script emits per-sample metadata (sample ID, render mode, GT availability)
- **THEN** metadata SHALL be logged under `world/info/*` while visual entities remain under `world/inputs/*`, `world/front/*`, and `world/side/*`
