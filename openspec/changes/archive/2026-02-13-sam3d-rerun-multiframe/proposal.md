## Why

The SAM-3D-Body rerun script currently visualizes only one frame per sample window, which limits temporal debugging and makes motion-level comparison difficult. Adding controlled multi-frame visualization is needed now to inspect sequence consistency and failure cases over time.

## What Changes

- Extend `scripts/visualize_sam3d_body_rerun.py` with a new CLI switch `--num-frames` to visualize multiple frames from the selected sample window.
- Keep existing `--frame-index` behavior for single-frame usage, and define deterministic interaction when both `--frame-index` and `--num-frames` are set.
- Update rerun timeline logging so each visualized frame is logged as a distinct timeline step.
- Ensure multi-frame mode works consistently across `overlay`, `mesh`, and `auto` render modes.
- Preserve existing namespace conventions (`world/inputs/*`, `world/front/*`, `world/side/*`, `world/info/*`).
- Non-goal: no changes to model training/evaluation logic and no change to dataset sampling semantics outside visualization.

## Capabilities

### New Capabilities
- `sam3d-rerun-multiframe-visualization`: Multi-frame sequence visualization in SAM-3D-Body rerun with explicit frame-count control.

### Modified Capabilities
- None.

## Impact

- Affected code:
  - `scripts/visualize_sam3d_body_rerun.py`
  - `docs/rerun_visualization.md`
- Affected modalities/datasets:
  - RGB temporal windows from config-selected dataset samples (HuMMan-oriented flow)
- Runtime output impact:
  - `.rrd` recordings under `logs/` will contain multiple timeline frames per sample when `--num-frames > 1`
- Compatibility:
  - Existing single-frame commands remain valid by default (`--num-frames` default of 1).
