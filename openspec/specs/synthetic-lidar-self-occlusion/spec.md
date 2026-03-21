# synthetic-lidar-self-occlusion Specification

## Purpose
Define regeneration and dataset integration requirements for self-occlusion-aware synthetic LiDAR artifacts.

## Requirements
### Requirement: Synthetic LiDAR Regeneration Shall Support Self-Occlusion-Aware Visibility
The system SHALL provide a `v1` synthetic LiDAR generation mode that reuses saved synthetic mesh and virtual LiDAR pose artifacts to generate a point cloud with self-occluded surfaces removed.

The `v1` mode SHALL use a depth-buffer visibility pass in the virtual LiDAR sensor frame and SHALL NOT rely only on face-normal filtering.

#### Scenario: Regenerate one sample with self-occlusion-aware visibility
- **WHEN** the user runs the `v1` LiDAR regeneration workflow on an accepted synthetic sample with saved mesh and virtual LiDAR metadata
- **THEN** the workflow SHALL write a regenerated point cloud that only contains points visible from the virtual sensor viewpoint within the configured depth-buffer visibility tolerance

### Requirement: Synthetic LiDAR Regeneration Shall Reuse Existing Synthetic Sample Artifacts
The `v1` LiDAR regeneration workflow SHALL operate on existing synthetic sample directories and SHALL NOT require rerunning person mask generation, SAM-3D-Body reconstruction, or target-format export.

#### Scenario: Update a saved synthetic sample without rerunning upstream stages
- **WHEN** the user runs the regeneration workflow on an existing synthetic dataset root
- **THEN** the workflow SHALL read the saved mesh, manifest, and virtual LiDAR pose artifacts directly and regenerate only the LiDAR-specific outputs and metadata

### Requirement: v1 LiDAR Artifacts Shall Coexist With Existing v0-a Artifacts
The `v1` regeneration workflow SHALL NOT overwrite the existing `v0-a` LiDAR artifact path. The sample SHALL retain both LiDAR versions, and the metadata SHALL allow downstream loaders to determine which artifact corresponds to which simulation version.

#### Scenario: Preserve both LiDAR versions in one sample
- **WHEN** the user regenerates a sample with the `v1` LiDAR workflow
- **THEN** the sample SHALL still retain the existing `v0-a` LiDAR artifact and SHALL also save the new `v1` LiDAR artifact with explicit version-identifying metadata

### Requirement: Regenerated Point Clouds Shall Carry Explicit Simulation Metadata
The regenerated `v1` point cloud outputs SHALL record explicit simulation metadata including the simulation mode or version, depth-buffer parameters, and the sampling counts before and after visibility filtering.

#### Scenario: Audit the source of a regenerated point cloud
- **WHEN** a user inspects a regenerated synthetic sample later
- **THEN** the sample metadata SHALL identify that the point cloud was produced by the self-occlusion-aware `v1` pipeline and SHALL record the core visibility parameters used to generate it

### Requirement: Dataset-Scale Regeneration Shall Be Resumable
The `v1` regeneration workflow SHALL support dataset-scale processing over an existing synthetic root and SHALL provide resumable progress tracking so interrupted runs can continue without reprocessing completed samples.

#### Scenario: Resume bulk LiDAR regeneration after interruption
- **WHEN** a dataset-scale `v1` regeneration run is restarted on a partially processed synthetic root
- **THEN** the workflow SHALL skip already completed samples according to saved progress or manifest state and continue with the remaining samples

### Requirement: The Workflow Shall Provide QC Comparison Support For v0-a And v1 Point Clouds
The system SHALL provide a way to visualize or summarize the difference between the original `v0-a` visible-surface point cloud and the regenerated `v1` self-occlusion-aware point cloud for selected samples.

#### Scenario: Compare old and new point clouds for validation
- **WHEN** the user runs the QC tooling on a regenerated synthetic sample
- **THEN** the tooling SHALL allow the user to inspect the previous and regenerated point clouds side by side with enough context to judge whether self-occluded surfaces were removed correctly

### Requirement: The Regeneration Workflow Shall Support Optional Inline QC Rendering
The `v1` regeneration workflow SHALL support an optional QC rendering mode that produces side-by-side validation outputs during regeneration and SHALL allow that rendering to be disabled for bulk dataset processing.

#### Scenario: Enable QC rendering only during validation
- **WHEN** the user runs the regeneration workflow on a small validation subset with QC rendering enabled
- **THEN** the workflow SHALL write QC comparison outputs for the selected samples

#### Scenario: Disable QC rendering during bulk regeneration
- **WHEN** the user runs dataset-scale regeneration with QC rendering disabled
- **THEN** the workflow SHALL process samples without writing the optional QC render outputs
