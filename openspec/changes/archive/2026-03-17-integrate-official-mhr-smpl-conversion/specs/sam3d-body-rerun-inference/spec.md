## ADDED Requirements

### Requirement: HuMMan SAM3D visualization SHALL support official converted SMPL comparison outputs
HuMMan-focused SAM3D visualization in MMHPE SHALL support a comparison mode that includes official converted SMPL outputs alongside raw SAM/MHR outputs and GT SMPL24 outputs. When enabled, the visualization path SHALL use the official conversion wrapper rather than a heuristic joint-remapping adapter.

#### Scenario: HuMMan comparison visualization exports converted SMPL outputs
- **WHEN** a user runs a HuMMan SAM3D visualization/export command that requests SMPL comparison outputs
- **THEN** the command SHALL include official converted SMPL results in the generated output set together with raw SAM/MHR and GT SMPL24 views

#### Scenario: Missing official conversion support blocks HuMMan comparison visualization
- **WHEN** a user requests HuMMan converted-SMPL visualization but the required official conversion dependencies are unavailable
- **THEN** the visualization command SHALL fail explicitly instead of silently falling back to heuristic joint remapping
