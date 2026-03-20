## ADDED Requirements

### Requirement: Synthetic Samples Can Be Exported To Multiple Training Target Formats
The project SHALL provide a synthetic export workflow that reads a generated SAM-3D-Body synthetic sample and writes target-format supervision bundles for at least HuMMan-compatible and Panoptic-compatible training.

The export workflow SHALL preserve the original base synthetic artifacts and SHALL add target-format outputs without replacing them.

#### Scenario: Export one accepted synthetic sample to HuMMan and Panoptic formats
- **WHEN** the user runs the target-format export workflow on an accepted synthetic sample
- **THEN** the workflow SHALL write HuMMan-compatible and Panoptic-compatible GT outputs and a manifest that identifies the target formats created

### Requirement: SMPL-Oriented Export Uses Upstream MHR Conversion
The synthetic export workflow SHALL support deriving SMPL-oriented supervision from SAM3D/MHR outputs by using the upstream MHR SMPL conversion path referenced by `facebookresearch/MHR/tools/mhr_smpl_conversion`.

The workflow SHALL record whether the conversion succeeded and SHALL preserve enough metadata to identify the conversion method and result quality.

#### Scenario: Convert synthetic MHR outputs into SMPL-oriented supervision
- **WHEN** the user requests HuMMan-compatible export for a valid synthetic sample
- **THEN** the workflow SHALL run the configured MHR-to-SMPL conversion path and save SMPL-oriented outputs or an explicit conversion failure status

### Requirement: HuMMan-Compatible Export Contract
The synthetic export workflow SHALL provide a HuMMan-compatible supervision bundle whose data contract matches the existing MMHPE HuMMan training-facing expectations for RGB plus LiDAR-style runs.

At minimum, the HuMMan-compatible export SHALL provide:
- SMPL24-style `gt_keypoints`
- `gt_smpl_params`
- `input_lidar`
- `rgb_camera`
- `lidar_camera`
- LiDAR-centered supervision variants required by current point-cloud transforms

#### Scenario: Export HuMMan-compatible LiDAR supervision
- **WHEN** a synthetic sample has valid SMPL-oriented conversion output and synthetic LiDAR data
- **THEN** the workflow SHALL save a HuMMan-compatible bundle with SMPL24 `gt_keypoints`, `gt_smpl_params`, camera metadata, LiDAR input, and LiDAR-centered GT variants

### Requirement: Panoptic-Compatible Export Contract
The synthetic export workflow SHALL provide a Panoptic-compatible supervision bundle with Panoptic COCO19 joint topology and Panoptic-compatible root/joint-order conventions.

The Panoptic-compatible export SHALL make the target topology explicit in metadata and SHALL save `gt_keypoints` in the Panoptic COCO19 order expected by current Panoptic training code.
The Panoptic-compatible export SHALL remain keypoint-only and SHALL NOT emit synthetic `gt_smpl_params`.

#### Scenario: Export Panoptic-compatible keypoints
- **WHEN** the user requests Panoptic-compatible export for an accepted synthetic sample
- **THEN** the workflow SHALL save `gt_keypoints` in Panoptic COCO19 order and record the topology name and root-joint convention in the export metadata

### Requirement: Export Avoids Replicated Dataset Trees And Unneeded Artifacts
The synthetic export workflow SHALL write only the minimal target-format arrays and metadata needed for downstream training use.

The workflow SHALL NOT create full HuMMan-like or Panoptic-like dataset folder trees and SHALL NOT replicate large source artifacts unless a specific exported target contract requires them.

#### Scenario: Export target GT without duplicating source data
- **WHEN** the user runs target-format export on an existing synthetic sample directory
- **THEN** the workflow SHALL write target-format supervision outputs without recreating a full dataset tree or duplicating unnecessary source artifacts

### Requirement: Coordinate-Space Variants Are Saved Explicitly
The synthetic export workflow SHALL save coordinate-space variants explicitly instead of overloading one unlabeled keypoint array.

For each exported target format, the workflow SHALL identify which saved arrays are in:
- canonical or new-world space
- camera space when exported
- LiDAR sensor space when exported
- PC-centered LiDAR space when exported

#### Scenario: Audit exported coordinate spaces
- **WHEN** a user inspects an exported synthetic sample later
- **THEN** the export manifest SHALL identify the skeleton topology and coordinate space of each GT array saved for that sample

### Requirement: Camera Metadata Remains Compatible With Existing MMHPE Transforms
The synthetic export workflow SHALL save per-modality camera metadata using the existing MMHPE camera contract of `intrinsic` plus `extrinsic`.

The workflow MAY also save derived `gt_camera_<modality>` pose encodings, but the raw camera dicts SHALL remain available so current transform pipelines can regenerate camera encodings and reprojected GT.

#### Scenario: Reuse exported cameras with camera transforms
- **WHEN** the exported synthetic sample is loaded by a training-facing dataset adapter or transform pipeline
- **THEN** the saved `rgb_camera` and `lidar_camera` metadata SHALL be sufficient to derive `gt_camera_<modality>` encodings with the existing camera transform utilities

### Requirement: Exported LiDAR GT Matches Current PC-Centering Contract
For exports that include `input_lidar`, the workflow SHALL save or derive GT arrays that are consistent with MMHPE's current LiDAR point-cloud centering contract.

At minimum, the workflow SHALL support a GT variant aligned with the PC-centered `input_lidar` frame used by existing point-cloud transforms.

#### Scenario: Export LiDAR GT for PC-centered training
- **WHEN** the workflow exports a sample with synthetic `input_lidar`
- **THEN** it SHALL save LiDAR GT that can be matched to the PC-centered LiDAR training contract without ambiguous frame conversion

### Requirement: Export Failure Is Explicit And Auditable
The synthetic export workflow SHALL fail fast or mark export status explicitly when a target-format export cannot be produced.

Examples include:
- MHR-to-SMPL conversion failure
- unsupported topology mapping
- missing camera metadata required by a target contract
- inconsistent LiDAR centering outputs

#### Scenario: Record target-format export failure
- **WHEN** one target export fails while processing a synthetic sample
- **THEN** the workflow SHALL save metadata identifying which target format failed and why, instead of silently omitting that output
