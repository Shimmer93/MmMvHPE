## Why

Current experiment configs under `configs/exp` use `seq_len = 1`, so dataset, transform, and model paths are mostly exercised in single-frame mode. This creates hidden risk for temporal runs (`seq_len > 1`) across MMHPE training/evaluation/prediction and blocks reliable multi-frame experiments for RGB, depth, LiDAR, and mmWave inputs.

## What Changes

- Audit the end-to-end pipeline for `seq_len > 1` handling, including config loading, dataset sampling, transforms, batching/collation, model forward paths, and relevant loss/metric interfaces.
- Fix modules that incorrectly assume `seq_len = 1` (shape flattening, indexing, temporal dimension ordering, or per-frame metadata handling).
- Add explicit validation/guardrails where behavior is unsupported so failures are early and actionable.
- Add or update `configs/exp` examples to exercise `seq_len > 1` runs.
- Add focused verification coverage (tests and/or reproducible run commands) for multi-frame behavior.
- Update `docs/` with sequence-shape expectations, constraints, and usage examples.

Scope boundaries and non-goals:
- In scope: correctness and compatibility for existing pipeline components when `seq_len > 1`.
- Out of scope: introducing new model architectures, changing dataset semantics, or broad performance optimization beyond correctness fixes.

## Capabilities

### New Capabilities
- `temporal-sequence-pipeline-compatibility`: Ensure dataset, transform, and model components correctly process `seq_len > 1` in config-driven MMHPE workflows.
- `temporal-sequence-validation`: Provide explicit validation and reproducible checks that catch temporal-shape incompatibilities before or during runs.

### Modified Capabilities
- None.

## Impact

- Affected code paths are expected in:
  - `configs/exp/` (sequence-length run definitions)
  - `datasets/` (sampling, item assembly, temporal metadata)
  - `datasets/*transform*` and related transform utilities
  - `models/` (encoder/aggregator/head/model API forward shape handling)
  - potentially `losses/`, `metrics/`, and `misc/` where temporal dimensions are consumed
- Runtime impact:
  - Existing `seq_len = 1` runs should remain backward compatible.
  - `seq_len > 1` runs should produce valid outputs/artifacts under `logs/` without shape-related failures from single-frame assumptions.
