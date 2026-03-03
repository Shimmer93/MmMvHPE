# config-aware-visualization-sampling Specification

## Purpose
TBD - created by archiving change temporal-seq2seq-output. Update Purpose after archive.
## Requirements
### Requirement: Config-Driven Model Input Windowing
Visualization scripts SHALL construct model input batches from dataset + pipeline configuration without overriding sequence/input semantics from CLI frame arguments.

#### Scenario: seq_len from config drives model input
- **WHEN** a demo config sets `train_dataset.params.seq_len: 1`
- **THEN** the visualization script SHALL feed one-frame windows to the model, even if `--num-frames` is greater than 1

#### Scenario: Split-specific dataset config is respected
- **WHEN** the user selects `--split train|val|test`
- **THEN** the script SHALL create the dataset using the corresponding split config block and pipeline block from the config file

### Requirement: Explicit Visualization-Time Frame Sampling
Visualization scripts SHALL use CLI arguments only to control timeline sampling for rerun logging and SHALL NOT reinterpret model prediction semantics.

#### Scenario: Within-window sampling for temporal windows
- **WHEN** the effective sample window has temporal length `T > 1` and `--num-frames K` is requested
- **THEN** the script SHALL log a contiguous frame sequence selected by `--frame-index` and bounded by window length `T`

#### Scenario: Cross-sample sampling for single-frame windows
- **WHEN** the effective sample window has temporal length `T = 1` and `--num-frames K` is requested
- **THEN** the script SHALL advance through `K` consecutive sample windows starting from `--sample-idx` and log one frame per window

### Requirement: Reproducible Temporal Metadata
Visualization scripts SHALL log enough metadata to reproduce the exact sample and frame selection used to build each rerun timeline step.

#### Scenario: Per-step metadata is available in rerun
- **WHEN** a frame is logged to rerun
- **THEN** the script SHALL log `world/info/sample_id` and `world/info/source_frame_index`

#### Scenario: Session-level temporal summary is logged
- **WHEN** a multi-frame visualization run starts
- **THEN** the script SHALL log total requested/actual frame count under `world/info/*`

