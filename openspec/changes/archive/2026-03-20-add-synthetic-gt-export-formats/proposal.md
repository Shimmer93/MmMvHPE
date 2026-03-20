## Why

The current SAM-3D-Body synthetic pipeline produces useful base artifacts, but it does not yet export supervision in the training-facing formats used by MMHPE's HuMMan and Panoptic pipelines. Without explicit target-format exports, the synthetic data cannot be used cleanly for model training or fair evaluation against those datasets.

## What Changes

- Add a synthetic GT export stage that derives target-format supervision bundles from the base SAM-3D-Body synthetic sample.
- Use the upstream MHR SMPL conversion tool to convert SAM3D/MHR outputs into SMPL or SMPL-X space, then derive SMPL24 joints for HuMMan-style training.
- Add a Panoptic-oriented export path that produces Panoptic COCO19 keypoints with the coordinate and root-joint conventions expected by Panoptic training configs.
- Export target-specific camera metadata and LiDAR-centered supervision variants so existing config-driven runs can reuse the same camera-head and point-cloud transforms.
- Keep the current base synthetic sample contract intact; target-format exports are additive and do not replace the saved MHR70 or mesh artifacts.
- Document scope boundaries: this change does not introduce a new training dataset into `main.py` yet and does not attempt to synthesize mmWave supervision.

## Capabilities

### New Capabilities
- `synthetic-target-format-export`: Export synthetic SAM-3D-Body samples into HuMMan-compatible and Panoptic-compatible GT bundles, including keypoint topology conversion, camera metadata, and LiDAR-centered supervision variants.

### Modified Capabilities
- None.

## Impact

- Affected code:
  - `projects/synthetic_data/`
  - synthetic-data CLI scripts under `scripts/`
  - new exporter utilities and format docs under `docs/`
- Affected runtime outputs:
  - additional per-sample arrays and metadata under synthetic output roots
  - optional format-specific export directories or manifests for HuMMan and Panoptic targets
- External dependency impact:
  - integration with the upstream MHR SMPL conversion tool described at `facebookresearch/MHR/tools/mhr_smpl_conversion`
- Config/runtime impact:
  - existing configs remain unchanged
  - exported artifacts are intended to match the current dataset and transform contracts used by config-driven MMHPE runs
