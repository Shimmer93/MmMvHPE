## Context

`scripts/visualize_inference_rerun.py` currently assumes a world-style visualization convention and applies fixed geometric flips/rotations before logging. That makes sensor-frame interpretation unclear when users want to inspect predictions against GT in the coordinate frame of a selected RGB/depth(lidar) sensor. The existing rerun helper stack already supports configurable view coordinates and camera transform helpers (`scripts/rerun_utils/session.py`, `scripts/rerun_utils/camera.py`), so this can be added incrementally without touching model training code.

## Goals / Non-Goals

**Goals:**
- Add an explicit coordinate-space mode for `visualize_inference_rerun.py`:
  - default: existing world-style behavior
  - new: sensor/camera-frame visualization
- Add sensor reference selection for multimodal/multiview samples (initial scope: RGB and depth/lidar camera metadata).
- Transform GT and predictions consistently into the selected sensor frame in sensor mode.
- Use camera-style rerun coordinates (Y-down) in sensor mode to avoid upside-down visualization.
- Log coordinate-space and selected reference-frame metadata under `world/info/*`.

**Non-Goals:**
- No changes to `main.py`, model forward/losses, or training/evaluation behavior.
- No dataset format or preprocessing changes.
- No mmWave frame-alignment support in this first pass.
- No redefinition of prediction semantics; only visualization mapping changes.

## Decisions

1. **CLI-first control with backward-compatible defaults**
   - Add switches in `scripts/visualize_inference_rerun.py` for coordinate mode and reference sensor/view.
   - Keep current behavior as default so existing commands/configs are unaffected.

2. **Reuse shared camera utility layer**
   - Use/extend `scripts/rerun_utils/camera.py` for extracting view extrinsics and applying transforms.
   - Avoid duplicating transform math in script body.

3. **Coordinate handling split by mode**
   - `world` mode: preserve current rotation path.
   - `sensor` mode: transform GT/pred using selected sensor extrinsic and bypass world-only flips to prevent double transforms.

4. **Explicit rerun coordinate declaration**
   - In sensor mode call `init_world_axes(rr.ViewCoordinates.RIGHT_HAND_Y_DOWN)`.
   - In world mode keep Y-up.

5. **Deterministic frame/view selection**
   - For multiview, bind transforms to explicit selected view index; fail fast on missing/mismatched camera metadata.
   - For temporal samples, apply the same frame-selection rules currently used by the script.

## Risks / Trade-offs

- **[Risk] Mixed coordinate transforms produce mirrored/misaligned outputs** -> Mitigation: strict branch separation (`world` vs `sensor`) and central transform helpers.
- **[Risk] Camera metadata mismatch in multiview samples** -> Mitigation: explicit validation with clear error messages (`num views` vs `num extrinsics`).
- **[Risk] Users assume sensor mode works for all modalities** -> Mitigation: document first-pass scope (RGB/depth-lidar) and fail explicitly for unsupported selections.
- **[Trade-off] More CLI complexity** -> Mitigation: safe defaults and concise help text.

## Migration Plan

1. Add coordinate-space/reference-frame CLI arguments to `scripts/visualize_inference_rerun.py`.
2. Integrate sensor-frame transform branch for GT and predictions.
3. Wire rerun view-coordinate selection by mode.
4. Add metadata logging (`world/info/coord_space`, `world/info/reference_sensor`, `world/info/reference_view`).
5. Add docs and runnable examples in `docs/rerun_visualization.md`.
6. Run smoke checks with one HuMMan config in both modes and verify `.rrd` output.

Rollback:
- Revert to default world-mode behavior by using defaults only; code path remains isolated.

## Open Questions

- Resolved: sensor mode requires explicit reference selection; no implicit default sensor/view.
- Resolved: expose `lidar` label for depth-derived point cloud reference to keep user-facing naming consistent with current pipeline.
