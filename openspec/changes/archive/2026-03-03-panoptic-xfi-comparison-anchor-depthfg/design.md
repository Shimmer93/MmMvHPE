## Context

This change spans dataset loading (`datasets/panoptic_preprocessed_dataset_v1.py`) and experiment configuration (`configs/`).

Current state:
- Panoptic preprocessed loader already supports temporal split and optional foreground-depth loading via `use_foreground_depth`.
- Panoptic configs exist for `PanopticHPE` training, and HuMMan configs exist for `XFi`, but there is no Panoptic-side `XFi` comparison suite.
- HuMMan dataset supports `anchor_key` semantics (for example, `input_rgb`) but Panoptic preprocessed loader does not yet expose equivalent behavior.

Constraints:
- Keep default Panoptic dataset behavior unchanged for existing configs.
- Keep configuration changes explicit and reproducible with sequence allowlists.
- Keep implementation minimal and fail-fast on invalid anchor/depth inputs.

## Goals / Non-Goals

**Goals:**
- Add optional anchor-coordinate conversion in Panoptic preprocessed dataset to support RGB-anchored GT for XFi.
- Add two Panoptic training configs (PanopticHPE vs XFi) with matched data settings:
  - temporal split,
  - one RGB + one LiDAR per sample,
  - foreground-depth loading,
  - non-piano sequence allowlist.
- Add four test configs for occluded/unoccluded sequence subsets across both models.

**Non-Goals:**
- No model architecture changes to PanopticHPE or XFi modules.
- No changes to Panoptic preprocessing outputs or sync-map generation.
- No denoising/filtering pipeline changes for depth or point clouds.

## Decisions

1. Add `anchor_key` to `PanopticPreprocessedDatasetV1` as an optional dataset parameter.
- Rationale: aligns with existing HuMMan interface and keeps anchoring as dataset concern.
- Alternatives considered:
  - Add a pipeline transform for anchoring: rejected, because anchoring requires coordinated updates to cameras and GT fields before downstream transforms.
  - Add model-side anchor handling: rejected, would couple data frame conventions to each model.

2. Implement anchoring by transforming camera extrinsics and GT keypoints after sample assembly and after optional new-world conversion.
- Rationale: preserves existing feature flow and ensures anchoring applies to final coordinate convention seen by training.
- Alternatives considered:
  - Anchor before new-world conversion: rejected as ambiguous ordering and harder to reason about with existing `apply_to_new_world` flag.

3. Use sequence allowlists directly in configs for subset control (non-piano, occluded, unoccluded).
- Rationale: explicit and auditable in config files; avoids hidden split behavior changes.
- Alternatives considered:
  - Add new split mode for these subsets: rejected as unnecessary complexity.

4. Put comparison configs under `configs/exp/panoptic/model_comparison/`.
- Rationale: keeps experiment suite discoverable without modifying existing baseline files.

## Risks / Trade-offs

- [Risk] Anchor mode with unsupported modality combinations can produce invalid assumptions.
  → Mitigation: strict validation of `anchor_key` against active sample modalities; explicit errors.

- [Risk] Sequence allowlists can drift from available dataset folders.
  → Mitigation: rely on existing dataset strict validation to fail fast when allowlist entries are missing.

- [Risk] XFi output shape mismatch if keypoint head still assumes 24 joints.
  → Mitigation: set Panoptic XFi config keypoint head output to `19 * 3` and use Panoptic skeleton format in visualization config fields.

## Migration Plan

1. Patch dataset class with optional anchor transform behavior.
2. Add six new Panoptic comparison configs.
3. Update Panoptic docs with new dataset parameters and config usage examples.
4. Validate by loading each config and fetching one sample from the dataset where applicable.

Rollback strategy:
- Revert the dataset anchor patch and remove new config files. Existing configs are unaffected because defaults preserve current behavior.

## Open Questions

- Whether future comparison runs should enforce colocated RGB/LiDAR pairing (`colocated: true`) for stricter sensor alignment. Current change keeps baseline Panoptic behavior unless explicitly configured.
