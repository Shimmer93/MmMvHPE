## Why

Current `PCMPJPE`/`SMPL_PCMPJPE` behavior removes both pelvis translation and pelvis orientation mismatch before error computation. For current evaluation needs, we want a translation-only variant so root rotation mismatch remains part of the reported error.

## What Changes

- **BREAKING** Change `PCMPJPE` and `SMPL_PCMPJPE` semantics to remove only pelvis translation before MPJPE.
- Remove pelvis orientation alignment from metric computation for both non-SMPL and SMPL paths.
- Update tests so pure-rotation offsets no longer collapse to near-zero error.
- Update docs to describe translation-only behavior and remove orientation-alignment claims.

Scope boundaries / non-goals:
- No model architecture, loss, or dataset preprocessing changes.
- No metric name changes; existing config references stay valid.
- No fallback behavior for malformed inputs.

## Capabilities

### New Capabilities
- None.

### Modified Capabilities
- `pelvis-centered-pose-metrics`: Redefine PCMPJPE/SMPL_PCMPJPE as pelvis-translation-centered MPJPE without root-orientation alignment.

## Impact

- Affected code: `metrics/mpjpe.py`, `metrics/smpl_metrics.py`, tests under `tests/`.
- Affected docs: `docs/pcmpjpe_metrics.md`.
- Runtime/log impact: metric values may increase versus prior behavior when root rotation differs.
