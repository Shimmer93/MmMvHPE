## MODIFIED Requirements

### Requirement: Standardized Rerun Logging Namespaces
The visualization pipeline SHALL log inputs, outputs, and metadata in standardized namespaces shared across scripts to support side-by-side comparison and tooling. It SHALL support both `world` and `sensor` coordinate modes for 3D visualization scripts and SHALL expose mode context in metadata.

#### Scenario: Metadata and render-mode info are logged consistently
- **WHEN** a visualization script emits per-sample metadata (sample ID, render mode, GT availability)
- **THEN** metadata SHALL be logged under `world/info/*` while visual entities remain under `world/inputs/*`, `world/front/*`, and `world/side/*`

#### Scenario: Coordinate mode metadata is logged for 3D inference scripts
- **WHEN** a script runs in `world` or `sensor` coordinate mode
- **THEN** it SHALL log coordinate-mode metadata under `world/info/*` for downstream interpretation of `.rrd` outputs
