## Why

The current MMHPE pipeline assumes a single camera per modality per sample, which blocks multi-view usage patterns where one modality (RGB, depth, LiDAR, or mmWave) must provide multiple synchronized camera streams. Enabling multiple cameras per modality is needed now to support upcoming `configs/exp` experiments while preserving existing one-camera workflows.

## What Changes

- Add support for multiple cameras per modality in the `configs/exp` execution path, including dataset sample structure, collate behavior, model inputs, and head/metric consumption paths that currently assume one camera per modality.
- Define and implement a backward-compatible input contract so existing one-camera configs continue to run without edits.
- Update only the required dataset/transform/model code paths used by `configs/exp` (not all legacy/dev configs).
- Add validation coverage for mixed cases:
  - one-camera-per-modality (legacy behavior)
  - multi-camera-per-modality for selected modalities
- Update docs with the new config schema and tensor/data-shape expectations for multi-camera modality inputs.

Scope boundaries and non-goals:
- In scope: minimum necessary code and config updates to run `configs/exp` with multi-camera-per-modality inputs.
- Out of scope: broad refactors for all non-`configs/exp` pipelines, introducing new model families, or performance optimization unrelated to correctness/compatibility.

## Capabilities

### New Capabilities
- `multi-camera-per-modality-exp-pipeline`: Allow each modality to carry multiple cameras in `configs/exp` runs and propagate them correctly through dataset, transforms, model forward, and evaluation paths.

### Modified Capabilities
- None.

## Impact

- Affected areas are expected in:
  - `configs/exp/` (new/updated multi-camera modality config patterns)
  - `datasets/` and `datasets/transforms/` (sample assembly, camera metadata, temporal/multi-camera collation)
  - `models/model_api.py`, aggregators, and heads used by `configs/exp` (feature extraction and prediction paths that currently assume one camera per modality)
  - `metrics/` for camera/keypoint consumers relying on one camera stream per modality
  - `docs/` for usage and shape contracts
- Runtime/log impact:
  - Existing one-camera `configs/exp` runs must remain functional and produce the same output structure under `logs/`.
  - Multi-camera `configs/exp` runs should produce valid predictions/metrics without shape or indexing failures from single-camera assumptions.
