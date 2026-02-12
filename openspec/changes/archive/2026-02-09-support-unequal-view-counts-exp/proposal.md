## Why

The previous multi-camera update enables per-modality views, but parts of the `configs/exp` pipeline may still implicitly assume equal view counts across modalities. That assumption is invalid: each modality must be allowed to have its own view count.

This change audits the same scope as the previous update and fixes only code paths that violate the unequal-view-count contract.

## What Changes

- Audit `configs/exp` runtime paths for hidden assumptions that all modalities share the same number of views.
- Enforce the contract: each modality has its own view count `V_m`.
- Apply targeted fixes only where violations are found; do not touch unaffected code.

Required behavior constraints:

- In `models/aggregators/trans_aggregator_v4.py`, preserve view information per modality instead of collapsing it too early.
- For modality-local attention (for example single attention), tokens for modality `m` should be represented as:
  - shape: `B, T, V_m * N_m + N_s, C`
  - where `V_m` is modality `m` view count, `N_m` is single-view token count for modality `m`, and `N_s` is special-token count.
- In keypoint heads, per-modality keypoint losses must be computed per view (not only on view-averaged targets/predictions).
- Only when computing the final global skeleton output should multi-view features be averaged across views.

Compatibility requirements:

- Keep one-camera configs fully backward compatible.
- Keep equal-view-count multi-camera configs working.
- Support unequal cross-modality view counts in the same batch/run.

Scope boundaries and non-goals:

- In scope: `configs/exp` paths, including datasets/transforms/model API/aggregator/heads/metrics/visualization touched by the previous change.
- Out of scope: unrelated refactors, architecture redesign, and code outside this scope unless strictly needed for correctness.

## Capabilities

### New Capabilities

- `unequal-cross-modality-view-counts-exp`: allow `rgb/depth/lidar/mmwave` to use different view counts in `configs/exp` runs without shape coupling errors.

### Modified Capabilities

- `multi-camera-per-modality-exp-pipeline`: refine behavior so view handling is modality-local, with per-view supervision and late view averaging only for final global skeleton computation.

## Impact

- Expected touched areas (only if violating assumption is present):
  - `models/aggregators/trans_aggregator_v4.py`
  - keypoint heads used by `configs/exp` (per-view loss behavior)
  - any related `models/model_api.py`, dataset/transform, metric, or visualization logic in the same scope
- Testing impact:
  - add focused tests for unequal view counts across modalities
  - add focused tests for per-view keypoint losses and late global view averaging behavior
- Runtime/log impact:
  - existing one-camera and equal-view-count runs remain stable
  - unequal-view-count runs no longer fail due to same-view-count assumptions.
