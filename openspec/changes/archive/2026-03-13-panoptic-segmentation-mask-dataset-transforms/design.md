## Context

Panoptic preprocessed data is loaded through `datasets/panoptic_preprocessed_dataset_v1.py` and composed into training and evaluation datasets via `datasets/data_api.py` and the transform registry under `datasets/transforms/`. The dataset already loads `input_rgb` and `input_depth` from per-sequence directories and records sequence identity, selected cameras, and frame identity in the sample. Separately, SAM3 person masks are now being generated under each sequence as `sam_segmentation_mask/<camera>/<frame>.png`.

The storage constraint is the main driver for this change. Materializing masked RGB and masked depth as new dataset trees would duplicate the Panoptic dataset and consume too much disk. The runtime pipeline is already config-driven, so the simplest integration point is dataset transforms that consume existing masks and modify loaded tensors in memory.

Constraints:
- The current training and visualization pipelines expect standard sample keys such as `input_rgb`, `input_depth`, `rgb_camera`, and `depth_camera`.
- Panoptic samples can be single-view or multi-view, and can be single-frame or short temporal windows.
- The project prefers fail-fast behavior over broad fallback logic.
- Documentation should live in `docs/`, not as long inline code comments.
- Depth masking must respect the Kinect RGB-depth baseline; direct 2D reuse of the RGB mask in the depth plane is not geometrically correct.

## Goals / Non-Goals

**Goals:**
- Add a runtime transform that applies existing Panoptic SAM3 masks to RGB and depth without creating new dataset copies.
- Keep RGB masking and depth masking independently configurable through one transform argument so experiments can enable either one or both.
- Resolve masks deterministically from sequence name, selected camera, and frame identity already present in the sample.
- Preserve tensor shapes and dtypes expected by the rest of the pipeline, with masked-out pixels set to zero.
- Keep the implementation incremental and aligned with the current dataset/transform architecture.

**Non-Goals:**
- Generating masks. That is already handled by the separate SAM3 mask-generation script.
- Supporting other datasets in the first version. This change is Panoptic-specific.
- Adding optional soft fallbacks for missing masks, alternate prompts, or mask fusion strategies.
- Writing masked RGB or depth images back to disk.

## Decisions

### 1. Implement one Panoptic-specific masking transform with `apply_to`
Add one transform under `datasets/transforms/` with an argument such as `apply_to=[rgb, depth]`. The transform will touch only the modalities explicitly requested and present in the sample. This keeps config intent explicit while avoiding duplicated mask loading logic.

Why this over two separate transforms:
- It still satisfies the proposal requirement that RGB and depth be independently configurable.
- It lets one transform load, decode, and validate the mask once when both modalities are active.
- It keeps configs shorter and avoids duplicated path-resolution logic in the pipeline.

Alternative considered:
- Two separate transforms for RGB and depth. Rejected because they would duplicate mask lookup and decoding work and complicate configs when both modalities need masking.

### 2. Resolve masks from dataset-root-relative sequence layout, not from ad hoc path parsing
The transforms should build mask paths from a strict contract:
- sequence root under the Panoptic preprocessed dataset root
- `sam_segmentation_mask/<camera>/<frame>.png`

The dataset already provides enough information to support this: `seq_name`, `selected_cameras`, `sample_id`, and the synchronized body-frame id encoded in the sample identity. The design should prefer explicit sample metadata over brittle filename parsing wherever possible.

Why this over scanning directories or best-effort matching:
- Mask lookup must be deterministic and cheap at runtime.
- Directory scans per sample would add avoidable I/O overhead.
- Best-effort matching would violate the project’s fail-fast principle.

Alternative considered:
- Add a preprocessing index file for masks. Rejected because it adds another artifact to maintain when the folder structure already provides direct addressing.

### 3. Apply masks before normalization/formatting and after frame loading
The transforms should run after the dataset has loaded raw `input_rgb` / `input_depth` arrays and before transforms such as normalization, tensor conversion, or model-specific formatting.

Why:
- Binary masking is easiest and least ambiguous in the raw image domain.
- Zeroing masked pixels before normalization preserves clean semantics; background becomes normalized zeros relative to the configured transform behavior.
- Applying masks after tensor formatting would force each transform to handle more layouts and dtypes.

Alternative considered:
- Apply masks after normalization. Rejected because it complicates value semantics and makes debugging visual outputs harder.

### 4. Use direct RGB-plane masking for RGB and reprojected masking for depth
For RGB, the transform should apply the loaded mask directly in the RGB image plane.

For depth, the transform should not apply the same 2D bitmap directly. Instead it should:
- back-project depth pixels into 3D with `K_depth`
- transform those 3D points from depth camera to color camera
- project them with `K_color`
- sample the RGB mask in the color image plane
- keep only depth pixels whose reprojected color samples land on foreground

This uses the calibrated RGB-depth viewpoint relation while keeping the transform cheap enough for runtime use.

Why:
- RGB masks live in the RGB image plane, not the depth image plane.
- even with corrected depth extrinsics, direct 2D mask reuse produces edge artifacts from parallax and self-occlusion.
- depth maps are small enough that a vectorized reprojection is practical.

Why:
- Zero fill is simple, deterministic, and consistent with existing padding behavior elsewhere in the repository.
- Depth already commonly uses zero as invalid / missing background.
- This preserves shape and avoids introducing new channels or metadata.

Alternative considered:
- Directly applying the same 2D RGB mask to depth. Rejected because it ignores the RGB-depth viewpoint difference.
- NaN for depth or alpha-masked RGB. Rejected because it would ripple into downstream transforms and model code.

### 5. Fail fast on missing, unreadable, or shape-mismatched masks
The default behavior should raise explicit errors when:
- the expected mask file does not exist
- the mask cannot be decoded
- the mask spatial shape does not match the target frame
- the sample does not expose enough metadata to resolve the mask path

Why:
- Silent skips would make experiments hard to trust.
- Shape mismatches likely indicate bad preprocessing or camera-name drift.
- The repository’s engineering rule is to raise explicit errors on abnormal inputs.

Alternative considered:
- A permissive fallback that leaves the frame unchanged. Rejected because it hides dataset inconsistencies and breaks experiment reproducibility.

### 6. Support sequence and multiview windows by applying one mask per frame per selected camera
Panoptic samples can contain temporal windows and potentially multiple selected views. The transform logic should iterate over the existing frame structure and apply the corresponding mask to each loaded frame for each selected camera in the sample.

Why:
- This keeps the change compatible with current Panoptic temporal configurations.
- It aligns with how `PanopticPreprocessedDatasetV1` already loads synchronized frame windows.
- The reprojection path remains cheap at Panoptic depth resolution.

Alternative considered:
- Restrict the first version to `seq_len=1` or single-view only. Rejected because the current Panoptic pipelines already use short sequences and camera sampling.

### 7. Document usage with Panoptic config examples rather than changing training entrypoints
No changes are needed in `main.py` or model APIs. The integration point is purely in dataset transforms and YAML configs. Documentation should be added under `docs/` with concrete examples showing how to enable RGB-only masking, depth-only masking, and combined masking.

Why:
- The current config-driven dataset pipeline already supports this extension model.
- Keeping the change out of training entrypoints reduces risk.

## Risks / Trade-offs

- [Extra per-sample disk I/O] -> Mitigation: use direct path resolution with no directory scans and decode each binary mask only once per active frame/camera pair even when masking both RGB and depth.
- [Depth reprojection adds per-sample compute] -> Mitigation: keep the implementation fully vectorized and reuse cached camera metadata and decoded masks.
- [Mask path resolution may break if camera naming is inconsistent between dataset samples and mask folders] -> Mitigation: define and enforce one canonical camera-name mapping in the transform implementation and raise explicit errors on mismatch.
- [Temporal and multiview frame structures are easy to mishandle] -> Mitigation: implement against the existing Panoptic sample layout and validate on both `seq_len=1` and multi-frame configs.
- [Zeroing background may change normalization statistics] -> Mitigation: make masking an explicit config choice and document that it changes the model input distribution.
- [Panoptic-only implementation may limit reuse] -> Mitigation: keep the first version strict and local to Panoptic; generalization can be proposed later if there is real demand.

## Migration Plan

- Add the new transform under `datasets/transforms/` and export it through `datasets/transforms/__init__.py`.
- Update Panoptic configs that want masked RGB and/or masked depth by inserting the new transform before normalization/formatting steps and setting `apply_to` accordingly.
- Add documentation in `docs/` with expected folder layout, failure modes, and command/config examples.
- Validate on one known sequence with existing SAM3 masks before broader training use, including masked/unmasked point-cloud comparison for depth.
- Rollback is trivial: remove the transforms from the config and the dataset returns to the current unmasked behavior.

## Open Questions

- None at the design level. The intended behavior is strict enough to proceed to the spec and implementation.
