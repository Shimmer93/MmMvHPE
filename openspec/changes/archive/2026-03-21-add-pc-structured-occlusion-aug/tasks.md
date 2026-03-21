## 1. Runtime Transform

- [x] 1.1 Add a structured point-cloud occlusion transform under `datasets/transforms/` for `input_lidar`.
- [x] 1.2 Implement range-image blob dropout and expose configurable probability and blob parameters.
- [x] 1.3 Verify that the transform preserves the downstream sample contract expected when it runs before centering and padding transforms.

## 2. Config Integration

- [x] 2.1 Add example YAML pipeline snippets or experiment configs that enable the generic range-image blob transform.
- [x] 2.2 Validate the transform order before `PCCenterWithKeypoints` against synthetic pretrain configs and at least one real-data LiDAR config.

## 3. Validation And Documentation

- [x] 3.1 Add usage documentation in `docs/` with `uv run` examples where relevant.
- [x] 3.2 Run loader-level validation to confirm augmented samples remain batchable in the target training pipelines.
- [x] 3.3 Document recommended conservative default parameters and ablation knobs for the generic blob transform.
