## Why

The Panoptic dataset now has reusable SAM3 person masks, but duplicating masked RGB or depth assets on disk would consume too much storage. Config-driven dataset transforms are needed so MMHPE can apply those masks at load time for training and evaluation without creating a second copy of the dataset.

## What Changes

- Add dataset pipeline transforms that read existing Panoptic SAM3 masks and apply them to loaded RGB and depth inputs at runtime.
- Support independent masking of RGB and depth so configs can enable either modality or both.
- Define fail-fast behavior for missing or mismatched mask files, with clear expectations for mask path resolution from sequence, camera, and frame identity.
- Add documentation and config usage examples showing how to enable the transforms in Panoptic dataset pipelines.

## Capabilities

### New Capabilities
- `panoptic-segmentation-mask-runtime-transforms`: Apply sequence-local SAM3 masks to Panoptic RGB and depth samples during dataset pipeline execution without writing masked copies to disk.

### Modified Capabilities
- None.

## Impact

- Affected components: Panoptic dataset loading, dataset pipeline transforms, Panoptic experiment configs, and documentation.
- Affected modalities: RGB and depth. LiDAR, mmWave, and other modalities are not directly changed in this proposal.
- Runtime impact: slightly higher per-sample load-time work and extra mask-file reads, but significantly lower storage usage compared with materialized masked datasets.
- Outputs remain config-driven runtime tensors; this change does not add new preprocessed dataset directories under `logs/` or under the dataset root.
