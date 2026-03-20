# sam3d-synthetic-sample-generation Specification

## Purpose
TBD - created by archiving change add-sam3d-synthetic-data-v0a. Update Purpose after archive.

## Requirements
### Requirement: End-to-End Synthetic Sample Generation
The project SHALL provide a `v0-a` synthetic sample generation workflow that starts from a single-image RGB human dataset sample and produces one saved synthetic sample artifact set using SAM-3D-Body reconstruction and one virtual LiDAR viewpoint.

The initial supported source dataset for this workflow SHALL be COCO val under `/opt/data/coco`.

The workflow SHALL treat `v0-a` as producer-side generation only and SHALL NOT require integration into `main.py` training or existing MMHPE dataset loaders.

For `v0-a`, the workflow SHALL keep SAM-3D-Body inference in the original image frame and SHALL NOT require a crop-first preprocessing path.

#### Scenario: Generate one synthetic sample from a valid COCO val image
- **WHEN** the user runs the `v0-a` synthetic generation workflow on a valid COCO val image containing an accepted human target
- **THEN** the workflow saves a synthetic sample artifact set containing at least source-image metadata, reconstructed 3D human outputs, one virtual LiDAR definition, one synthetic point cloud, and any enabled optional visualization outputs

#### Scenario: Reject unsupported or unusable source input
- **WHEN** the selected input sample does not satisfy the workflow’s acceptance rules for `v0-a` (for example no valid target person, invalid crop, or failed reconstruction)
- **THEN** the workflow SHALL fail fast or mark the sample rejected with an explicit reason instead of silently producing incomplete synthetic outputs

### Requirement: Explicit Intermediate Stage Outputs
The synthetic workflow SHALL preserve stage-level outputs and metadata for the following stages:
- source image and target-person selection,
- full-image mask generation and saving,
- SAM-3D-Body reconstruction,
- quality filtering,
- virtual LiDAR sampling,
- point-cloud synthesis,
- visualization.

The saved metadata SHALL be sufficient to determine which stage failed or rejected the sample.

#### Scenario: Trace a rejected sample to a pipeline stage
- **WHEN** a sample is rejected during generation
- **THEN** the saved metadata SHALL identify the rejection stage and rejection reason

#### Scenario: Inspect intermediate artifacts for an accepted sample
- **WHEN** a sample is accepted by the workflow
- **THEN** the saved artifact set SHALL contain enough intermediate outputs to inspect saved-mask quality, reconstruction quality, and point-cloud synthesis quality

### Requirement: Saved Full-Image Person Mask
The workflow SHALL save a person mask for the selected human target in the original source-image frame.

The saved mask SHALL remain traceable to the source image without requiring an intermediate crop coordinate system.

If the source dataset provides a valid person segmentation for the selected target, the workflow SHALL be able to use that segmentation as the saved mask. If the source segmentation is missing or unusable, the workflow MAY use a generation fallback, but it MUST still save the resulting full-image mask artifact and provenance metadata.

#### Scenario: Save source-frame person mask
- **WHEN** the workflow accepts a source sample for processing
- **THEN** it SHALL save a person mask artifact aligned to the original image frame and record the mask provenance in metadata

### Requirement: SAM-3D-Body Auxiliary Mask Input
The workflow SHALL support passing the saved person mask into SAM-3D-Body as auxiliary input for `v0-a` reconstruction.

The workflow SHALL keep the original image frame and SHALL NOT require a crop-first inference path in order to use the mask.

#### Scenario: Run reconstruction with saved mask input
- **WHEN** the workflow runs SAM-3D-Body inference for an accepted sample
- **THEN** it SHALL provide the saved person mask as auxiliary input to reconstruction and record that mask-assisted mode in the sample metadata

### Requirement: Canonical 3D Supervision Contract
The workflow SHALL produce canonical 3D human supervision outputs intended for later MMHPE reuse.

At minimum, the workflow SHALL save 3D keypoints in a pelvis-centered canonical frame and SHALL record the transform or metadata needed to relate those outputs back to the reconstruction/sensor frame used during generation.

The workflow SHALL keep coordinate-frame handling explicit and SHALL NOT silently mix canonical outputs with raw reconstruction-frame outputs.

#### Scenario: Save canonical and traceable 3D outputs
- **WHEN** the workflow saves 3D supervision outputs for an accepted sample
- **THEN** the sample metadata SHALL identify the canonical frame used for saved keypoints and how it relates to the source reconstruction frame

### Requirement: Virtual LiDAR Point-Cloud Synthesis
The workflow SHALL support one sampled virtual LiDAR sensor pose per `v0-a` sample and SHALL synthesize one LiDAR-style point cloud from the reconstructed human body surface as viewed from that sensor.

For `v0-a`, the project MUST support visible-surface point sampling from the virtual sensor viewpoint. A beam-accurate LiDAR simulator is not required in this phase.

The saved synthetic point cloud SHALL include enough metadata to reproduce the sensor pose and sampling parameters.

#### Scenario: Save one synthetic LiDAR point cloud
- **WHEN** reconstruction succeeds and the sample passes quality filtering
- **THEN** the workflow SHALL save one synthetic point cloud and the corresponding virtual LiDAR pose metadata for that sample

#### Scenario: Preserve future extensibility beyond v0-a simulation fidelity
- **WHEN** the `v0-a` workflow synthesizes a point cloud
- **THEN** it SHALL record the simulation mode and parameters so later milestones can distinguish simple visible-surface sampling from more realistic LiDAR simulation modes

### Requirement: Optional Visual Quality-Control Outputs
The workflow SHALL support saving visualization outputs for qualitative inspection of generated synthetic samples.

When enabled, the visual outputs SHALL cover at least:
- source RGB image,
- saved full-image person mask,
- SAM-3D-Body mesh or body reconstruction overlay,
- reconstructed 3D keypoints,
- virtual LiDAR pose/context,
- synthetic point cloud rendering.

#### Scenario: Review a generated sample visually
- **WHEN** visualization saving is enabled and the user opens the saved visualization outputs for an accepted sample
- **THEN** the outputs SHALL allow the user to assess whether the selected person, 3D reconstruction, and synthetic point cloud are qualitatively plausible

### Requirement: Reproducible Generation Metadata
The workflow SHALL save reproducibility metadata for each processed sample.

This metadata SHALL include:
- source dataset identity and sample/image reference,
- mask provenance,
- SAM-3D-Body checkpoint/runtime identity,
- virtual LiDAR sampling parameters,
- point-cloud synthesis mode and parameters,
- output artifact paths or identifiers,
- acceptance or rejection status.

#### Scenario: Reproduce an accepted sample artifact
- **WHEN** a user revisits a saved accepted sample later
- **THEN** the metadata SHALL provide enough information to understand how the sample was generated and which core parameters/checkpoints were used

#### Scenario: Audit a rejected sample
- **WHEN** a sample is rejected
- **THEN** the metadata SHALL still be saved with rejection status and rejection reason so the pipeline behavior can be audited
