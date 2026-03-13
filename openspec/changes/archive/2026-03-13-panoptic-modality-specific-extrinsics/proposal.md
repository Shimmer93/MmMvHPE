## Why

The current Panoptic preprocessed dataset loader collapses RGB, depth, and depth-derived LiDAR to the same `extrinsic_world_to_color` transform whenever that field is present in preprocessed camera metadata. On this machine that path is active for all preprocessed Panoptic sequences, which means depth and LiDAR samples are not using their modality-specific extrinsics even though the metadata contains distinct `M_color` and `M_depth` transforms.

## What Changes

- Update Panoptic preprocessed dataset loading so RGB uses color extrinsics while depth and depth-derived LiDAR use depth extrinsics by default.
- Define the precedence rules between `extrinsic_world_to_color` and modality-specific metadata so the runtime behavior is explicit and testable.
- Validate and document the expected Panoptic camera-metadata contract for modality-specific extrinsic loading.
- Add validation and visualization checks that confirm RGB/depth/LiDAR samples use the intended sensor geometry.
- **BREAKING**: Panoptic depth and depth-derived LiDAR samples may change sensor-frame coordinates compared with current runs that shared RGB extrinsics.
- Explicitly leave RGB-mask-to-depth reprojection or rectification out of scope for this change; that will be handled as a separate follow-up if needed.

## Capabilities

### New Capabilities
- None.

### Modified Capabilities
- `panoptic-preprocessed-dataset-v3-interface`: Panoptic samples must expose modality-appropriate camera extrinsics instead of reusing RGB/world-to-color extrinsics for depth and depth-derived LiDAR.

## Impact

- Affected code: `datasets/panoptic_preprocessed_dataset_v1.py` and any Panoptic-specific visualization or evaluation utilities that interpret sample camera extrinsics.
- Affected modalities: RGB, depth, and depth-derived LiDAR for the Panoptic preprocessed dataset. mmWave and non-Panoptic datasets are out of scope.
- Affected configs: existing Panoptic configs do not need new keys, but runs using depth or LiDAR may produce different sensor-frame results after the fix.
- Runtime/log impact: training losses may shift for depth/LiDAR pipelines, and sensor-frame visualizations or debug exports under `logs/` may move because the underlying extrinsics become modality-correct.
- Non-goal for this change: improving RGB/depth mask alignment by reprojection. Correct extrinsics are a prerequisite for that later work, but they are not the same feature.
