## Context

Current SAM-3D-Body rerun visualization in `scripts/visualize_sam3d_body_rerun.py` compares predictions in camera coordinates against HuMMan V3 GT that is typically provided in canonical pelvis-centered space (`apply_to_new_world=True` in dataset flow). This is a coordinate-frame mismatch, not a model error, and it can visually bias qualitative analysis.

The dataset path already provides per-view camera extrinsics (`rgb_camera`) through `HummanPreprocessedDatasetV3` (inherited from `HummanPreprocessedDatasetV2`). The missing piece is an explicit, deterministic conversion path in visualization code and metadata that records which GT frame is used.

## Goals / Non-Goals

**Goals:**
- Add explicit GT coordinate selection for SAM-3D-Body visualization: `canonical` or `camera`.
- Convert GT keypoints/GT keypoint sequence to camera coordinates using selected RGB view extrinsics when `camera` is requested.
- Keep behavior deterministic across single-view and multiview execution paths.
- Record selected coordinate space in rerun metadata/log output for traceability.
- Keep existing script entrypoints and config-driven workflow intact.

**Non-Goals:**
- No changes to model training (`main.py`, `models/model_api.py`) or losses/metrics.
- No changes to dataset storage format or preprocessing outputs.
- No changes to SAM-3D-Body inference output frame definition.
- No depth/LiDAR/mmWave coordinate conversion in this change.

## Decisions

1. **Coordinate conversion lives in script helper layer, not dataset classes**
   - Decision: implement GT canonical->camera conversion in `scripts` helper modules used by `visualize_sam3d_body_rerun.py`.
   - Rationale: this is a visualization concern; changing dataset outputs would affect all training/eval pipelines and risk regressions.
   - Alternative considered: add new dataset output fields (`gt_keypoints_cam`, `gt_keypoints_seq_cam`). Rejected for now to avoid widening the data contract.

2. **Use extrinsic matrix directly: `X_cam = R * X + t`**
   - Decision: use each selected view's `extrinsic` from `sample["rgb_camera"]` to transform GT 3D keypoints.
   - Rationale: this is consistent with existing projection assumptions in the codebase and avoids introducing additional calibration dependencies.
   - Alternative considered: invert transforms or depend on world-space reconstruction utilities. Rejected because GT input for this script is canonical, and direct forward transform is sufficient.

3. **Add explicit config/CLI switch with config as default source**
   - Decision: support `gt_coordinate_space` in visualization config, and a CLI override for quick checks.
   - Rationale: config remains reproducible; CLI enables fast debugging.
   - Alternative considered: auto-infer from dataset flags. Rejected because it is implicit and error-prone in mixed experiments.

4. **Temporal consistency: transform per frame and per selected view**
   - Decision: for `gt_keypoints_seq`, apply conversion for every frame; for multiview, apply the matching camera for each view.
   - Rationale: prevents static/misaligned GT artifacts in multiframe rerun output.
   - Alternative considered: convert only center frame. Rejected because it breaks temporal visualization guarantees.

5. **Document behavior in `docs/` and annotate rerun recording**
   - Decision: update documentation and add coordinate-space tag in the output record/metadata.
   - Rationale: avoids ambiguity when comparing archived `.rrd` files.

## Risks / Trade-offs

- **[Risk] Multi-view camera indexing mismatch** -> Mitigation: centralize view-to-camera resolution in one helper and validate camera count vs rendered views with explicit errors.
- **[Risk] Missing `rgb_camera` field in some samples/configs** -> Mitigation: fail fast for `camera` mode with actionable message; allow `canonical` mode to proceed.
- **[Risk] Ambiguity between camera/world/canonical naming** -> Mitigation: enforce only two accepted user-facing values (`canonical`, `camera`) and document semantics.
- **[Trade-off] Script-level conversion duplicates some geometry logic** -> Mitigation: keep conversion helper small and reusable; do not fork dataset behavior.
- **[Trade-off] No backward compatibility requirement for ambiguous legacy flags** -> Mitigation: keep existing defaults unchanged unless the new option is set.

## Migration Plan

1. Add `gt_coordinate_space` option to SAM-3D visualization config(s) under `configs/demo/`.
2. Implement GT conversion helpers under `scripts/` helper directory and wire into `scripts/visualize_sam3d_body_rerun.py`.
3. Update rerun logging metadata to include selected GT coordinate space.
4. Run smoke test on HuMMan V3 with `canonical` and `camera` modes; verify side-view qualitative alignment.
5. Update `docs/` with usage examples and coordinate-space definitions.

Rollback strategy:
- Revert to `gt_coordinate_space=canonical` default and bypass conversion helper paths if issues are found.

## Open Questions

- Resolved: visualization stays single-mode (one GT coordinate space per run) for clarity.
- Resolved: generalization to other visualization scripts is a later follow-up; this change targets SAM-3D-Body rerun only.
