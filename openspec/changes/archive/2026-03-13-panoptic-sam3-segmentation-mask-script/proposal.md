## Why

The Panoptic preprocessing workflow now depends on reliable foreground masks, but the dataset does not ship with SAM3-style segmentation masks in the per-sequence layout expected by this project. A dedicated generation script is needed so Panoptic RGB frames can be segmented consistently with the SAM-3D-Body pipeline and stored once for reuse during preprocessing and evaluation.

## What Changes

- Add a script under `tools/` that generates person segmentation masks for Panoptic sequence RGB images using the same SAM3 segmentation behavior used by SAM-3D-Body.
- Support sequence-local output layout: `<sequence>/sam_segmentation_mask/<kinect_camera>/<image_name>` with one binary mask per RGB image.
- Use the text prompt `person` and union all person instances in the frame into a single binary mask.
- Allow the script to run on selected sequences and cameras so partially downloaded or partially processed Panoptic datasets can be handled incrementally.
- Add documentation covering required checkpoints/dependencies, expected directory layout, and concrete `uv run python ...` usage examples.

## Capabilities

### New Capabilities
- `panoptic-sam3-segmentation-mask-generation`: Generate per-image binary person masks for Panoptic RGB frames using SAM3-style prompting and save them into each sequence directory in a deterministic camera-specific layout.

### Modified Capabilities
- None.

## Impact

- Affected data/components: Panoptic dataset sequence directories, RGB modality assets, SAM3-based tooling under `tools/`, and documentation under `docs/`.
- Runtime outputs will be written into dataset storage under each sequence’s `sam_segmentation_mask/` tree rather than into `logs/`.
- Dependencies will rely on the existing SAM3 / SAM-3D-Body environment already used elsewhere in the repository; the new script will need to validate checkpoint availability and fail fast when that environment is incomplete.
