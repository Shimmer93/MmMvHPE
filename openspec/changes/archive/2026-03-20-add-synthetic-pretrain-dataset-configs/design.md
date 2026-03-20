## Context

The synthetic SAM-3D-Body pipeline already produces per-sample export bundles under each synthetic sample directory. Those bundles are intentionally storage-efficient: they reuse the source COCO image path, save only the mask once, and write minimal HuMMan-style and Panoptic-style supervision payloads under `exports/`.

MMHPE's training stack, however, still only knows how to read real-dataset layouts such as HuMMan and Panoptic preprocessing outputs. The existing staged workflows rely on config-driven datasets under `datasets/`, stage-1 HPE configs, stage-2 camera-head configs, and final fixed-lidar-frame evaluation through `tools/eval_fixed_lidar_frame.py`.

The synthetic-to-real experiment needs to fit this existing architecture with minimal churn:
- synthetic pretraining must look like a normal dataset run from `main.py`
- stage-1 and stage-2 configs must remain close to the real-data configs the team already uses
- final evaluation must stay on the real datasets and keep using the fixed-lidar-frame tooling

The main constraint is that Panoptic synthetic exports intentionally do not replicate HuMMan-like LiDAR/camera payload files. The dataset layer therefore has to reconstruct the missing LiDAR-side Panoptic supervision from the base synthetic sample artifacts instead of expanding the on-disk export format.

## Goals / Non-Goals

**Goals:**
- Add one training-facing synthetic dataset class that reads exported synthetic samples directly from the sample-centric folder structure.
- Support both HuMMan-style SMPL24 supervision and Panoptic-style joints19 supervision from the same synthetic root.
- Keep the synthetic dataset compatible with the existing train pipelines and transforms, especially `CameraParamToPoseEncoding`, `PCCenterWithKeypoints`, `VideoNormalize`, and `PCPad`.
- Provide staged config sets for:
  - synthetic stage-1 pretraining
  - synthetic stage-2 camera-head pretraining
  - real-data stage-1 finetuning initialized from synthetic stage-1 checkpoints
  - real-data stage-2 finetuning initialized from real stage-1 checkpoints and optionally synthetic stage-2 camera-head checkpoints
  - final evaluation on real HuMMan and Panoptic datasets
- Document the expected synthetic-root structure and end-to-end training/evaluation commands.

**Non-Goals:**
- Re-export the synthetic dataset into full HuMMan or Panoptic directory trees.
- Change the synthetic exporter to duplicate source RGBs or add large new artifact payloads.
- Replace the current real-dataset loaders.
- Introduce a new training loop or bypass `main.py`.

## Decisions

### Decision: Use one dataset class with a `target_format` switch

The implementation will add a single synthetic dataset class under `datasets/` that accepts `target_format: humman|panoptic`.

Rationale:
- both formats share the same sample-root discovery, image loading, and point-cloud loading logic
- the format-specific differences are limited to GT topology, camera payload selection, and a small amount of on-the-fly derived supervision
- one class reduces duplicated loader code and keeps config authoring simpler

Alternatives considered:
- separate `SyntheticHummanDataset` and `SyntheticPanopticDataset`
  - rejected because most logic is identical and would drift over time
- modifying `HummanPreprocessedDatasetV2` / `PanopticPreprocessedDatasetV1`
  - rejected because the synthetic sample layout is fundamentally different from the real preprocessed dataset layouts

### Decision: Keep the sample-centric synthetic folder layout as the source of truth

The loader will read:
- the base sample manifest
- the format-specific export manifest under `exports/humman` or `exports/panoptic`
- the original source RGB via `image_path`
- the synthetic LiDAR points from the base sample artifacts or linked export file

Rationale:
- the user explicitly wants to avoid replicated files and full dataset-tree mirroring
- the existing export format already contains the minimum authoritative metadata needed to reconstruct training samples

Alternatives considered:
- flattening all samples into a single training manifest and duplicating payload files
  - rejected because it increases storage and introduces a second source of truth

### Decision: Return stage-1 keys in the same contract as real datasets

The synthetic dataset will emit the same core keys used by the real staged pipelines:
- `sample_id`
- `modalities`
- `input_rgb`
- `input_lidar`
- `rgb_camera`
- `lidar_camera`
- `gt_keypoints`
- `gt_smpl_params` for HuMMan-style samples
- `gt_global_orient`
- `gt_pelvis`
- `gt_keypoints_2d_rgb`
- `gt_keypoints_lidar`
- `gt_keypoints_pc_centered_input_lidar`
- `selected_cameras`

Rationale:
- this keeps the existing transform stacks and heads reusable
- stage-2 camera-head configs can consume the synthetic dataset directly instead of going through JSON skeleton side channels

Alternatives considered:
- a reduced synthetic-only contract with separate transforms
  - rejected because it would create a parallel training path and make comparison with real configs harder

### Decision: Derive missing Panoptic LiDAR-side supervision on the fly

Panoptic exports remain keypoint-only on disk, but the dataset will derive:
- `lidar_camera`
- `gt_camera_lidar`
- `gt_keypoints_lidar`
- `gt_keypoints_pc_centered_input_lidar`

from:
- base synthetic `lidar_extrinsic_world_to_sensor`
- base synthetic `synthetic_lidar_points_sensor`
- base synthetic `pelvis_source_frame`
- exported Panoptic `gt_keypoints_world`
- exported Panoptic `gt_keypoints`
- exported Panoptic `gt_pelvis`

The derivation will map the Panoptic new-world frame into the existing synthetic LiDAR sensor frame without writing extra files.

Rationale:
- preserves the user's "minimal export only" constraint
- still gives the stage-2 Panoptic camera-head path the LiDAR-aligned supervision it needs

Alternatives considered:
- extending the exporter to save Panoptic LiDAR camera bundles for every sample
  - rejected for now because it duplicates data already derivable from the base manifest

### Decision: Synthetic pretraining configs use `rgb` + `lidar`, not `rgb` + `depth`

The synthetic dataset will expose LiDAR directly, so the synthetic configs will use:
- `modality_names: ['rgb', 'lidar']`

Rationale:
- the synthetic pipeline already produces point clouds, not depth maps
- it avoids forcing a fake depth modality only to immediately convert it back to LiDAR

Alternatives considered:
- pretending the synthetic LiDAR is a depth-derived modality
  - rejected because it obscures the actual input contract and adds unnecessary transform assumptions

### Decision: Finetune configs stay as copies of the existing real-data configs with explicit synthetic checkpoint initialization

The new config sets will be added alongside new synthetic pretrain configs. The real-data finetune/eval configs will stay close to the existing HuMMan and Panoptic references, only changing:
- experiment naming
- checkpoint initialization paths
- stage naming/comments for the synthetic-to-real workflow

Rationale:
- the goal is an apples-to-apples synthetic pretrain ablation, not a second modeling branch
- keeping the real configs close to the current references lowers the review burden

Alternatives considered:
- deeply refactoring the existing config hierarchy around inheritance
  - rejected because the repo does not currently use a dedicated YAML inheritance system for these experiments

## Risks / Trade-offs

- [Panoptic LiDAR derivation mismatch] → Compute the LiDAR-side Panoptic supervision from the same base manifest artifacts used during synthetic generation, and validate sample shapes and finite values when loading.
- [RGB path drift if synthetic roots are moved] → Fail fast when `image_path` no longer exists and document that the synthetic dataset assumes the source COCO root remains available.
- [Transform incompatibility with seq_len > 1 assumptions] → Restrict the synthetic dataset to `seq_len=1` and raise explicit errors for unsupported temporal settings.
- [Synthetic stage-2 overfitting to GT skeleton inputs] → Keep synthetic stage-2 configs explicit about using GT 2D/3D supervision and position them as pretraining only; real stage-2 finetuning remains the primary target.
- [Config sprawl] → Group the new configs under a dedicated synthetic-transfer directory and keep them named by style and stage.

## Migration Plan

1. Add the synthetic dataset class and register it in `datasets/__init__.py`.
2. Add a documentation page for the dataset contract and the synthetic-to-real training workflow.
3. Add synthetic pretrain configs for HuMMan-style and Panoptic-style stage 1 and stage 2.
4. Add real-data finetune and final-eval configs that reference the synthetic-pretrained checkpoints.
5. Smoke-test dataset instantiation and config loading on the exported synthetic root.

Rollback is simple because the change is additive:
- remove the synthetic dataset class
- remove the new config directory
- remove the synthetic training docs

## Open Questions

- None. The export-policy questions were resolved by the user:
  - no replicated HuMMan/Panoptic dataset trees
  - Panoptic on-disk export stays keypoint-only
  - internal conversion target is SMPL
  - HuMMan exports keep `gt_smpl_params`
