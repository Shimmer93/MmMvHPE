## Why

XFi modality handling is currently implicit and tightly coupled to hard-coded modality ordering in the aggregator. This causes runtime failures when enabled backbones and dataset modalities differ (for example RGB+LiDAR without depth/mmWave), and makes configuration-driven experiments brittle.

## What Changes

- Add config-driven modality selection for XFi aggregation so active modalities are explicitly declared and validated.
- Define deterministic modality ordering used by XFi aggregation and projector branch mapping.
- Add fail-fast validation for invalid modality declarations and missing required inputs.
- Update XFi-oriented configs to declare active modalities explicitly.
- Add documentation for modality configuration and common valid combinations (for example RGB+depth, RGB+LiDAR).
- Non-goal: changing model math, loss definitions, or introducing new sensor modalities.

## Capabilities

### New Capabilities
- `xfi-configurable-modalities`: Configure which modalities XFi aggregation consumes from YAML and enforce strict runtime validation of modality availability and ordering.

### Modified Capabilities
- None.

## Impact

- Affected code: `models/aggregators/xfi_aggregator.py`, `configs/baseline/unseen_view_generalization/humman_xfi.yml`, and `docs/` for usage guidance.
- API/config impact: new/required aggregator config fields for active modality list.
- Runtime impact: clearer aggregation-time failures for invalid modality combinations instead of late `NoneType` errors; no change to `logs/` artifact locations.
