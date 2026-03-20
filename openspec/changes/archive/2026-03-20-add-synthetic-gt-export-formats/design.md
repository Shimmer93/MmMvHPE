## Context

The existing `v0-a` synthetic pipeline reconstructs a human body from a single RGB image with SAM-3D-Body, saves an MHR70-style 3D skeleton and mesh, samples one virtual LiDAR pose, and saves a synthetic point cloud. That output is useful for inspection, but MMHPE training does not consume raw MHR70 supervision directly.

The current dataset contracts are split by target dataset family:

- HuMMan-style training expects SMPL-oriented `gt_keypoints`, `gt_smpl_params`, per-modality camera dicts, and LiDAR-centered variants such as `gt_keypoints_pc_centered_input_lidar`.
- Panoptic-style training expects Panoptic COCO19 keypoints, Panoptic root conventions, and the same general `input_*`, `*_camera`, `gt_keypoints` contract.
- Camera heads derive `gt_camera_<modality>` from `*_camera` plus image size, so exporter outputs must preserve that information cleanly.

The user also pointed to the upstream MHR conversion utility in `facebookresearch/MHR/tools/mhr_smpl_conversion/README.md`. That tool explicitly supports converting SAM3D outputs in MHR space back to SMPL/SMPL-X parameters and meshes, which is a better foundation than creating a local mesh-to-joint regressor without provenance.

## Goals / Non-Goals

**Goals:**

- Add a synthetic export stage that turns one base synthetic sample into multiple target GT formats.
- Make SMPL24 the canonical intermediate format for HuMMan-oriented export.
- Make Panoptic COCO19 a first-class export target for Panoptic-oriented experiments.
- Preserve explicit coordinate-frame handling across canonical/new-world, camera, LiDAR, and PC-centered LiDAR spaces.
- Keep exports compatible with current dataset transforms such as `CameraParamToPoseEncoding` and `PCCenterWithKeypoints`.
- Document the exporter contract and the role of the upstream MHR conversion dependency.

**Non-Goals:**

- No direct integration into `main.py` training in this change.
- No attempt to export mmWave supervision.
- No attempt to synthesize full Panoptic or HuMMan folder trees; the exporter should write only the minimal target-format artifacts and metadata needed for training.
- No requirement to recover fully reliable SMPL pose and shape for every sample; failures may be filtered or marked explicitly.

## Decisions

### 1. Keep a two-layer contract: base synthetic sample first, target-format export second

The existing synthetic generator should remain the producer of source RGB, masks, MHR70 keypoints, mesh, and virtual-sensor artifacts. A separate export layer should derive HuMMan/Panoptic-compatible supervision from those saved artifacts.

Why:
- It keeps synthetic generation dataset-agnostic.
- It avoids coupling the generator to one training target.
- It makes re-export possible if skeleton mapping or camera conventions change later.

Alternative considered:
- Writing HuMMan/Panoptic-specific keys directly during initial generation.
  - Rejected because it mixes source-space reconstruction with training-target policy and makes future format evolution harder.

### 2. Use the upstream MHR conversion tool for SMPL recovery

The exporter should use the MHR SMPL conversion tool referenced by the user to convert SAM3D/MHR outputs into SMPL parameters/meshes. SMPL24 joints should then be derived from that converted representation, and the fitted SMPL parameters should be stored in the export outputs.

Why:
- The upstream tool is specifically documented for converting `SAM3D Outputs (MHR)` back into SMPL-family representations.
- It provides a traceable and supported path instead of a project-local approximation.
- It can return fitted parameters, meshes, and fitting errors, which are useful for quality gates.

Alternatives considered:
- Direct local mesh-to-SMPL24 joint regression.
  - Rejected for the first version because it is less auditable and likely less faithful.
- Fitting a custom SMPL model directly in MMHPE.
  - Rejected because it duplicates a tool that already exists upstream.

### 3. Export multiple skeleton topologies from one canonical intermediate

The exporter should store:
- raw MHR70
- SMPL24
- Panoptic COCO19

SMPL24 should be the main intermediate for additional downstream mappings. Panoptic COCO19 should be produced through an explicit mapping stage with documented joint order.

Why:
- HuMMan and many existing configs are SMPL-oriented.
- Panoptic training/evaluation expects a different topology and root index.
- Keeping both avoids repeated conversion work and makes validation easier.

Alternative considered:
- Export only one unified skeleton.
  - Rejected because the training code already depends on dataset-specific joint contracts.

### 4. Export coordinate-space variants explicitly instead of overloading `gt_keypoints`

For each relevant target format, the exporter should save:
- canonical/new-world keypoints
- camera-frame keypoints where needed
- LiDAR-frame keypoints
- PC-centered LiDAR keypoints

The export manifest should identify the topology and coordinate space of each saved array.

Why:
- Existing MMHPE transforms assume strict frame semantics.
- LiDAR-centered training is sensitive to mismatches between `input_lidar`, camera extrinsics, and centered GT.

Alternative considered:
- Save only `gt_keypoints` and recompute everything else implicitly.
  - Rejected because it hides assumptions and makes audits harder.

### 5. Reuse existing camera metadata contract rather than storing only encoded poses

The exporter should save `rgb_camera` and `lidar_camera` as modality camera dicts with `intrinsic` and `extrinsic`. `gt_camera_<modality>` pose encodings may be exported too, but they should remain derived data.

Why:
- That matches the dataset contract already consumed by current transforms.
- It keeps the export compatible with `CameraParamToPoseEncoding`.

Alternative considered:
- Save only 9D camera pose encodings.
  - Rejected because it loses the richer camera contract and makes reprojection/debugging harder.

### 6. Export minimal per-sample target bundles, not dataset-tree replicas

The exporter should write target-format arrays and manifests alongside or under the existing synthetic sample directory, without creating full HuMMan-like or Panoptic-like dataset trees or replicating large source artifacts.

Why:
- The user explicitly wants to avoid replicated files.
- The existing synthetic pipeline already stores the base sample once.
- The first exporter phase is for reusable supervision, not for mimicking raw dataset storage layouts.

Alternative considered:
- Emitting full HuMMan/Panoptic-like folder trees.
  - Rejected because it duplicates storage and couples the exporter to dataset-specific file-layout details that are not required for training-facing GT generation.

### 7. Panoptic export stays keypoint-only

The Panoptic export should save only the keypoint supervision and supporting metadata needed for Panoptic-oriented training. It should not emit placeholder `gt_smpl_params`.

Why:
- The user explicitly scoped Panoptic export to keypoints only.
- Placeholder SMPL data adds ambiguity without helping the target training path.

Alternative considered:
- Mirroring the current Panoptic loader's zero-valued `gt_smpl_params`.
  - Rejected for the exporter because the requested target format is keypoint-only and synthetic outputs should avoid unnecessary fields.

## Risks / Trade-offs

- [SMPL conversion quality may vary by pose, occlusion, or reconstruction failure] -> Save conversion status and fitting errors, and allow explicit rejection or partial export.
- [Upstream conversion tool may add environment or model-file requirements] -> Keep integration isolated in an adapter module and document prerequisites clearly.
- [Joint mappings may silently drift if the source topology changes] -> Make mapping tables explicit, versioned, and covered by small validation tests.
- [LiDAR-centered exports may mismatch current training transforms] -> Validate against `PCCenterWithKeypoints` expectations and compare exported centered arrays with transform-generated equivalents.
- [Panoptic and HuMMan root conventions differ] -> Store per-format topology metadata and root indices in manifests rather than assuming a single root convention.

## Migration Plan

1. Add an exporter specification and implementation under `projects/synthetic_data/`.
2. Integrate an adapter for the upstream MHR conversion tool.
3. Add format-specific exporters for SMPL24 and Panoptic COCO19.
4. Add validation scripts or smoke tests on already generated synthetic samples.
5. Document how to run export on top of an existing synthetic output directory.

Rollback is straightforward because the change is additive: stop calling the exporter and keep using the existing base synthetic sample outputs.
