## Why

We need an H36M multiview RGB setup and configuration to test FusionFormer under the paper’s setting (multi-view, multi-frame) and assess reproducibility. This enables a clearer comparison to reported results before extending to multimodal fusion.

## What Changes

- Add a multiview H36M dataset path that returns multiple camera views per sample for RGB.
- Add GT-derived 2D keypoints for H36M (from 3D camera-space joints) to stand in for a 2D estimator.
- Add a FusionFormer config for H36M with T=27, B=2, 17 joints, and all cameras enabled via config.

## Capabilities

### New Capabilities
- `h36m-multiview-rgb`: Dataset support for multi-camera RGB sequences per sample on H36M.
- `h36m-gt-2d-adapter`: GT-derived 2D keypoint adapter for H36M camera-space joints.
- `h36m-fusionformer-config`: H36M FusionFormer experiment config aligned with paper settings.

### Modified Capabilities
- (none)

## Impact

- Affected code: datasets (new H36M multiview dataset or extension), transforms (GT 2D adapter), configs (new H36M FusionFormer config).
- Training/eval workflows: new config for H36M FusionFormer; multiview RGB only; GT 2D used during training/eval.
- Modalities: RGB only; multi-camera views configurable in config.
