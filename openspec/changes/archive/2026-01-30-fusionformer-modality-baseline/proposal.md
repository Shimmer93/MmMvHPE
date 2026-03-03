## Why

We need a camera-parameter-free baseline that matches the FusionFormer paper while fitting Humman’s multimodal data layout. This provides a reliable reference point for RGB+depth/PC fusion before exploring more complex estimators or end-to-end pipelines.

## What Changes

- Add a FusionFormer-style model that treats modalities (RGB, depth/point cloud) as “views” and predicts 3D keypoints only.
- Add data plumbing to feed GT 2D (RGB) and GT 3D (PC) poses as stand-ins for off-the-shelf estimators.
- Add a minimal train/test-mini config to validate end-to-end execution on HummanPreprocessedDatasetV2.
- Document the baseline’s assumptions (camera-parameter-free, modality-as-view) and limits.

## Capabilities

### New Capabilities
- `fusionformer-modality-baseline`: FusionFormer baseline that fuses modality tokens over time and outputs 3D keypoints.
- `pose-estimator-gt-adapters`: Dataset/pipeline adapters that provide GT 2D/3D pose inputs in place of external estimators.

### Modified Capabilities
- (none)

## Impact

- Affected code: models (new FusionFormer module), datasets/pipelines (GT pose adapters), configs (new Humman mini config).
- Training/eval workflows: add a new baseline config and model name; training uses GT 2D/3D pose proxies; evaluation reports MPJPE/PAMPJPE on keypoints.
- Modalities: RGB + depth/point cloud fused as modality “views”; camera intrinsics ignored.
