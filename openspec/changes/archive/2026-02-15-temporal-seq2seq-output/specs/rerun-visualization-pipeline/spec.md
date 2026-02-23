## MODIFIED Requirements

### Requirement: Shared Visualization Core for Multiple Inference Backends
The system SHALL provide a shared visualization core for sample loading, model execution, timeline stepping, and rerun logging that can be reused by multiple inference adapters. The shared core SHALL separate (1) config-driven model input construction from (2) CLI-driven visualization frame selection so that scripts remain comparable while preserving model input contracts.

#### Scenario: Existing MMHPE script uses shared core without CLI breakage
- **WHEN** `scripts/visualize_inference_rerun.py` is executed with existing required arguments
- **THEN** it SHALL run through the shared visualization core and produce rerun outputs with the same timeline semantics for equivalent sample/frame selections

#### Scenario: Existing SAM3D script remains compatible with shared sampling rules
- **WHEN** `scripts/visualize_sam3d_body_rerun.py` is executed with `--sample-idx`, `--frame-index`, and `--num-frames`
- **THEN** frame-selection behavior SHALL match the shared core semantics used by the MMHPE inference script

#### Scenario: seq_len=1 visualization uses cross-sample stepping
- **WHEN** effective temporal length per sample is 1 and user requests `--num-frames > 1`
- **THEN** the shared core SHALL step across consecutive sample windows for timeline generation without changing how each sample is fed to the model

### Requirement: Standardized Rerun Logging Namespaces
The visualization pipeline SHALL log inputs, outputs, and metadata in standardized namespaces shared across scripts to support side-by-side comparison and tooling.

#### Scenario: Metadata and render-mode info are logged consistently
- **WHEN** a visualization script emits per-sample metadata (sample ID, render mode, GT availability)
- **THEN** metadata SHALL be logged under `world/info/*` while visual entities remain under `world/inputs/*`, `world/front/*`, and `world/side/*`

#### Scenario: Temporal selection metadata is logged consistently
- **WHEN** a visualization step is written to rerun
- **THEN** the script SHALL log `world/info/source_frame_index` and temporal run summary metadata under `world/info/*`
