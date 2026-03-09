## ADDED Requirements

### Requirement: Config-Driven Single-Camera SAM3 Panoptic Evaluation
The system SHALL provide a segment-evaluation entry script under a dedicated analysis directory in `tools/` that accepts a config file using the current Panoptic/SAM3 dataset style, a selected split (`train`, `val`, or `test`), and one explicit RGB camera identifier such as `kinect_008`. The script SHALL evaluate only that selected camera stream even when the config contains multiple RGB cameras.

The script SHALL remain compatible with existing config-driven runs from `configs/` by loading dataset and pipeline settings through the same config merge path currently used by SAM3 visualization scripts. The script MUST fail fast if the requested camera is not available in the selected split after dataset filtering.

#### Scenario: Select one camera from a multiview RGB config
- **WHEN** the user runs the analysis script with a multiview Panoptic RGB config, `--split test`, and `--camera kinect_008`
- **THEN** the script SHALL instantiate the dataset from the config, restrict analysis to `kinect_008`, and evaluate only samples belonging to that camera stream

#### Scenario: Requested camera is absent
- **WHEN** the selected split does not contain any samples for the requested camera after config and split filtering
- **THEN** the script SHALL terminate with an explicit error that names the missing camera and split

### Requirement: Deterministic Non-Overlapping Segment Construction
The system SHALL partition the selected `(sequence, camera)` sample stream into contiguous, non-overlapping temporal segments of one user-specified segment length per run. Supported segment lengths MUST include values such as `8`, `16`, and `32`, and the script SHALL accept exactly one segment length value in each invocation.

Segments SHALL NOT mix frames from different sequences or different cameras. If the tail of a `(sequence, camera)` stream contains fewer frames than the requested segment length, that incomplete tail SHALL be dropped and SHALL NOT be scored.

Each segment record MUST include at least:
- `sequence_name`
- `camera_name`
- `segment_index`
- `segment_length`
- `start_frame_id`
- `end_frame_id`
- `num_frames`
- `sample_indices`

#### Scenario: Build fixed-length segments within one sequence-camera stream
- **WHEN** the selected split contains a synchronized sample stream for one `(sequence, camera)` pair and the user passes `--segment-length 16`
- **THEN** the script SHALL emit consecutive 16-frame segments in dataset order without overlap

#### Scenario: Drop incomplete tail segment
- **WHEN** the final contiguous frame block in one `(sequence, camera)` stream has fewer frames than the requested segment length
- **THEN** the script SHALL exclude that block from metric computation and from the scored segment list

### Requirement: SAM3 Prediction Joints Must Be Adapted to Panoptic COCO19
The system SHALL convert SAM-3D-Body prediction joints into Panoptic COCO19 joint order before metric computation. The conversion SHALL be based on explicit SAM3 joint semantics, not on an undocumented positional assumption.

The adapter SHALL output a tensor/array with shape `(19, 3)` per analyzed frame in Panoptic COCO19 order:
- `nose`
- `neck`
- `mid_hip`
- `right_shoulder`
- `right_elbow`
- `right_wrist`
- `left_shoulder`
- `left_elbow`
- `left_wrist`
- `right_hip`
- `right_knee`
- `right_ankle`
- `left_hip`
- `left_knee`
- `left_ankle`
- `right_eye`
- `left_eye`
- `right_ear`
- `left_ear`

If a required Panoptic COCO19 joint cannot be mapped or derived unambiguously from SAM3 outputs, the system MUST fail fast with an actionable error instead of silently filling incorrect values.

#### Scenario: Successful named-joint adaptation
- **WHEN** SAM-3D-Body inference returns the expected named joint outputs for a frame
- **THEN** the adapter SHALL produce one `(19, 3)` Panoptic COCO19 prediction array for that frame before metric computation

#### Scenario: Required joint mapping is missing
- **WHEN** the SAM3 output metadata does not expose enough joint semantics to recover a required Panoptic COCO19 joint
- **THEN** the analysis run SHALL terminate with an error that identifies the missing target joint and the source mapping failure

### Requirement: Per-Segment MPJPE Family Evaluation
The system SHALL compute `MPJPE`, `PA-MPJPE`, and `PC-MPJPE` for every analyzed frame after adapting predictions to Panoptic COCO19, then aggregate those frame-level values into per-segment metrics.

For each scored segment, the system SHALL record at least:
- `mpjpe_mean`
- `pa_mpjpe_mean`
- `pc_mpjpe_mean`
- `mpjpe_max`
- `pa_mpjpe_max`
- `pc_mpjpe_max`
- `num_valid_frames`

The system SHALL reuse the project’s existing metric semantics for `MPJPE`, `PA-MPJPE`, and `PC-MPJPE` so that segment analysis remains comparable to other MMHPE evaluations.

#### Scenario: Compute three metrics for one complete segment
- **WHEN** a segment contains valid GT and SAM3 predictions for all frames
- **THEN** the script SHALL compute frame-level `MPJPE`, `PA-MPJPE`, and `PC-MPJPE` and write the aggregated segment metrics to the result log

#### Scenario: Invalid frame output is encountered
- **WHEN** a frame in a segment cannot produce a valid adapted `(19, 3)` prediction or GT comparison target
- **THEN** the script SHALL either exclude that frame from the segment metric count with explicit bookkeeping or fail the run according to a documented strictness mode, and SHALL NOT silently treat the frame as zero error

### Requirement: Structured Logs and Sequence-Camera Aggregation
The system SHALL write run outputs under `logs/` in a dedicated analysis run directory. Each run SHALL produce machine-readable segment logs and summary reports that make it possible to identify the worst-performing segments and the worst-performing `(sequence, camera)` pairs.

At minimum, the run directory MUST contain:
- a per-segment CSV or equivalent tabular log
- a per-segment JSON or equivalent structured log
- a ranked worst-segments summary
- an aggregated summary grouped by `(sequence_name, camera_name)`
- one or more plots showing metric distributions or worst-performing `(sequence, camera)` regions

The `(sequence_name, camera_name)` grouped summary SHALL aggregate segment statistics per pair and SHALL be part of the standard output, not an optional post-processing step.

#### Scenario: Write segment logs and grouped summary
- **WHEN** the script completes a run for one split, one camera, and one segment length
- **THEN** it SHALL write per-segment logs plus an aggregated `(sequence_name, camera_name)` summary into the run output directory under `logs/`

#### Scenario: Worst-performing segments can be ranked directly
- **WHEN** the user opens the generated CSV or JSON outputs
- **THEN** the outputs SHALL contain enough metadata and metric columns to sort segments by worst `MPJPE`, `PA-MPJPE`, or `PC-MPJPE`

### Requirement: Optional Worst-Segment Visualization Export
The system SHALL support an optional run mode that exports visual inspection artifacts for the worst-performing segments after metric computation. The user SHALL be able to request a top-K export count, and the workflow SHALL emit artifacts only for the K worst-ranked segments according to a selected metric.

The exported artifact format MAY be rerun recordings or another project-standard visualization artifact, but the format MUST preserve:
- sequence name
- camera name
- frame ids in the segment
- the metric used for ranking

If top-K export is not requested, the main metric evaluation workflow SHALL still complete normally without producing those extra visualization artifacts.

#### Scenario: Export top-K worst segments
- **WHEN** the user requests `--export-worst-k 10 --rank-metric mpjpe`
- **THEN** the script SHALL identify the 10 worst-scoring segments by `MPJPE` and write visualization artifacts for those segments in the run output directory

#### Scenario: Skip visualization export by default
- **WHEN** the user does not request worst-segment export
- **THEN** the script SHALL write metric logs and plots only, without generating extra visualization files
