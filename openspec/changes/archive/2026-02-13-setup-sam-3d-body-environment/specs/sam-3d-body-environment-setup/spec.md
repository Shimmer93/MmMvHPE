## ADDED Requirements

### Requirement: SAM-3D-Body environment setup SHALL be reproducible in MMHPE
The project SHALL provide a documented, repeatable environment setup path for `third_party/sam-3d-body` that is executable from the MMHPE repository using `uv` conventions and compatible with the existing Python/CUDA baseline used by MMHPE. SAM-3D-Body support SHALL be treated as a required repository environment capability, not an optional add-on.

#### Scenario: Setup instructions are executable from repo root
- **WHEN** a developer follows the documented setup commands from the repository root
- **THEN** the commands SHALL complete without requiring edits to source files outside the documented steps
- **THEN** the setup SHALL not require manual modification of `uv.lock`

#### Scenario: Existing MMHPE entrypoint remains runnable
- **WHEN** SAM-3D-Body environment setup is completed
- **THEN** existing config-driven MMHPE runs through `main.py` SHALL remain runnable without mandatory SAM-3D-Body flags

### Requirement: Checkpoint path contract SHALL be explicit and validated
The setup SHALL define `/opt/data/SAM_3dbody_checkpoints/` as the checkpoint root for SAM-3D-Body and SHALL validate existence/readability of required assets before runtime checks are reported as successful.

Required assets:
- `/opt/data/SAM_3dbody_checkpoints/model_config.yaml`
- `/opt/data/SAM_3dbody_checkpoints/model.ckpt`
- `/opt/data/SAM_3dbody_checkpoints/assets/mhr_model.pt`

#### Scenario: Required checkpoint assets are present
- **WHEN** `/opt/data/SAM_3dbody_checkpoints/model_config.yaml`, `/opt/data/SAM_3dbody_checkpoints/model.ckpt`, and `/opt/data/SAM_3dbody_checkpoints/assets/mhr_model.pt` all exist and are readable
- **THEN** environment validation SHALL pass the checkpoint preflight stage

#### Scenario: Any required checkpoint asset is missing or unreadable
- **WHEN** one or more required files in the checkpoint contract is missing or unreadable
- **THEN** environment validation SHALL fail with an actionable error message including the exact missing/unreadable file path

### Requirement: A smoke validation path SHALL verify SAM-3D-Body operability
The project SHALL provide a lightweight validation path (script and/or command sequence) that verifies SAM-3D-Body can be imported and initialized in the configured environment without running full benchmarking or training.

#### Scenario: Dependencies and imports are valid
- **WHEN** the smoke validation is executed in a configured environment
- **THEN** it SHALL verify required SAM-3D-Body module imports
- **THEN** it SHALL exit successfully only if imports and minimal initialization checks pass

#### Scenario: Dependency mismatch is detected
- **WHEN** required SAM-3D-Body dependencies are missing or incompatible
- **THEN** the smoke validation SHALL fail fast with a clear diagnostic that identifies the missing or incompatible dependency

### Requirement: GPU runtime SHALL be required for SAM-3D-Body setup validation
SAM-3D-Body setup validation SHALL assume GPU-capable runtime and SHALL fail explicitly when CUDA runtime/device requirements are not met.

#### Scenario: CUDA runtime is available
- **WHEN** the validation environment has accessible CUDA devices
- **THEN** GPU preflight checks SHALL pass and validation SHALL proceed

#### Scenario: CUDA runtime is unavailable
- **WHEN** no CUDA device is available or CUDA runtime is not usable
- **THEN** validation SHALL fail with a message indicating GPU runtime is required for SAM-3D-Body

### Requirement: Environment setup SHALL be isolated from benchmark integration behavior
This change SHALL only establish environment readiness and SHALL NOT introduce benchmark/demo execution behavior into MMHPE training/evaluation pipelines.

#### Scenario: No benchmark logic introduced in environment setup change
- **WHEN** this change is applied
- **THEN** there SHALL be no new requirement to run SAM-3D-Body during standard MMHPE train/val/test workflows unless explicitly invoked via dedicated setup validation commands
