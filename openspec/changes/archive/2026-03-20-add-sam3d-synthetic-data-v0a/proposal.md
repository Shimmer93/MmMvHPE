## Why

MMHPE is currently trained mainly on lab-collected datasets such as HuMMan and Panoptic/Kinoptic, where camera placement, orientation, and scene conditions are limited. That makes it difficult to study or improve generalization to in-the-wild viewpoints and sensor layouts using the current training data alone.

We need a first synthetic-data pipeline now to test whether single-image RGB datasets can be converted into MMHPE-compatible supervision by using SAM-3D-Body for human reconstruction and a virtual LiDAR sensor for point-cloud synthesis. A narrow `v0-a` milestone lets us validate the idea end-to-end before expanding to large-scale generation or full training integration.

## What Changes

- Add a new synthetic data generation subproject focused on a `v0-a` end-to-end demo pipeline.
- Use single-image RGB human data from COCO val under `/opt/data/coco` as the initial source dataset.
- Run a staged synthetic pipeline:
  - select/filter one-person RGB inputs,
  - obtain and save a person mask in the original image frame,
  - run SAM-3D-Body on the full image with the saved mask as auxiliary input to reconstruct mesh and 3D keypoints,
  - sample one virtual LiDAR sensor pose,
  - synthesize one LiDAR-style point cloud from the reconstructed body surface,
  - save sample artifacts and visual debugging outputs.
- Define the artifact/data contract for generated `v0-a` samples so later phases can export them into MMHPE-compatible dataset format.
- Add visualization and quality-control outputs so generated samples can be inspected before any training use.
- Keep scope intentionally narrow:
  - no large-scale batch generation requirement,
  - no training integration into `main.py` yet,
  - no realistic full LiDAR beam simulator yet,
  - no multi-frame or multi-person pipeline yet,
  - no crop-first preprocessing path in `v0-a`.

## Capabilities

### New Capabilities
- `sam3d-synthetic-sample-generation`: Generate and visualize small-scope synthetic RGB-to-3D-to-LiDAR samples from single-image RGB datasets using SAM-3D-Body and one virtual LiDAR viewpoint.

### Modified Capabilities

## Impact

- Affected code: new generation/visualization modules under a dedicated synthetic-data subproject area (for example `projects/` or `tools/`), plus supporting docs in `docs/`.
- Affected datasets/modalities: RGB source images first; generated outputs target RGB + synthetic LiDAR point cloud + 3D keypoints. Depth and mmWave are out of scope for `v0-a`.
- Affected dependencies/systems: relies on the existing `third_party/sam-3d-body` environment and checkpoint contract; uses COCO val from `/opt/data/coco` as the initial source dataset.
- Affected runtime outputs: synthetic sample artifacts, metadata, and visualization outputs under `logs/` or a dedicated synthetic output root.
- Scope boundary: this change establishes the first generation pipeline and saved sample contract only; it does not yet modify MMHPE training/evaluation pipelines or commit to a final large-scale synthetic dataset format.
