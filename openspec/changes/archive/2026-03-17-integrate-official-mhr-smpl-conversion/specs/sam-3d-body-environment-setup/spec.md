## ADDED Requirements

### Requirement: Official MHR conversion dependencies SHALL be validated for HuMMan conversion workflows
The SAM-3D-Body environment contract SHALL define and validate the additional runtime dependencies required for official MHR-to-SMPL conversion workflows used by HuMMan evaluation and visualization. At minimum, the environment validation SHALL fail explicitly when the `mhr` package or required SMPL model asset for the selected conversion target is unavailable.

#### Scenario: Official MHR conversion dependencies are available
- **WHEN** a user runs a HuMMan SAM3D workflow that requests official MHR-to-SMPL conversion and the required conversion dependencies and model assets are installed and readable
- **THEN** environment validation SHALL permit the workflow to continue

#### Scenario: Official MHR conversion dependency or asset is unavailable
- **WHEN** a user runs a HuMMan SAM3D workflow that requests official MHR-to-SMPL conversion and the `mhr` package or required SMPL model asset is missing or unreadable
- **THEN** the workflow SHALL fail fast with an actionable error message that identifies the missing dependency or asset path
