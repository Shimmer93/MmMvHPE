## Why

Our current FusionFormer feature extractor does not match the paper’s formulation, which
expects a per‑joint embedding followed by a pose feature extractor that models joint
relationships. We need to align the extractor to the paper and implement the EFC baseline
(3‑layer FCN) for a faithful comparison.

## What Changes

- Add an EFC (3‑layer FCN) feature extractor that maps per‑joint embeddings to pose features.
- Wire the EFC extractor into FusionFormer so pose features match the paper’s definition
  \(F^{(0)}_{pose} \in \mathbb{R}^{T \times V \times C_P}\).
- Update configs to select the EFC baseline extractor for experiments.

## Capabilities

### New Capabilities
- `fusionformer-efc-extractor`: Provide a 3‑layer FCN pose feature extractor baseline aligned to the paper.

### Modified Capabilities
- 

## Impact

- Affected code: `models/` (feature extractor), FusionFormer aggregator wiring, and configs.
- Affected workflows: training/eval when selecting the EFC baseline extractor.
- Modalities: applies to pose‑based inputs (RGB 2D pose; depth/point cloud if used similarly).
