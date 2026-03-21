## Why

Even with self-occlusion-aware synthetic point clouds, the saved dataset still lacks the structured missing regions caused by other objects or sensor dropouts in real scenes. A runtime augmentation layer is the right place to add that variability because it can produce many occlusion patterns from the same saved synthetic or real LiDAR sample without forcing a new frozen dataset version for every augmentation policy.

## What Changes

- Add one generic runtime point-cloud augmentation transform for range-image blob occlusion during training.
- Remove contiguous regions in range-image space rather than isolated random points.
- Make the augmentation configurable from existing dataset pipelines in YAML configs.
- Add training-facing documentation and example config snippets for enabling or disabling the augmentation in synthetic pretrain and synthetic-to-real finetune experiments.

## Capabilities

### New Capabilities
- `pc-structured-occlusion-augmentation`: generic runtime range-image blob point-cloud occlusion augmentation for LiDAR inputs in config-driven training pipelines.

### Modified Capabilities
- None.

## Impact

- Affected code: point-cloud transforms under `datasets/transforms/`, synthetic-transfer configs under `configs/`, and usage docs under `docs/`.
- Affected runs: synthetic pretraining and finetuning experiments can enable the augmentation without regenerating dataset artifacts.
- Runtime impact: training data loading will become slightly more expensive when the augmentation is enabled.
- Output impact: no new dataset files are required; behavior changes only inside the training pipeline configured from YAML.
