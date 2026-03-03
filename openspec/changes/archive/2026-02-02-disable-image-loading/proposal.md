## Why

Training runs that only use keypoints are still loading and holding image data in H36M multiview, which wastes memory and slows throughput. Adding a way to disable image loading for H36MMultiViewDataset reduces memory pressure and enables larger batches or longer sequences without changing model outputs.

## What Changes

- Add a configuration option to disable image loading in H36MMultiViewDataset when images are not used.
- Ensure the H36MMultiViewDataset pipeline can skip image decoding/augmentation while preserving keypoint and metadata loading.
- Update documentation/examples to show how to run keypoints-only training with image loading disabled.

## Capabilities

### New Capabilities
- `image-loading-toggle`: Allow datasets/pipelines to skip loading image data via configuration while still loading keypoints and required metadata.

### Modified Capabilities
- 

## Impact

- Affected code: H36MMultiViewDataset, its data pipeline/transforms, and config files used for H36M training/eval.
- Affected workflows: keypoints-only training/evaluation; multimodal runs remain unchanged unless the option is set.
- Dependencies/APIs: configuration schema changes for relevant datasets/pipelines.
