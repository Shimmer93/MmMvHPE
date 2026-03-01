## Why

`scripts/visualize_inference_rerun.py` currently visualizes 3D outputs in a fixed world-style convention, which makes sensor-frame interpretation (for RGB/depth/lidar multiview runs) ambiguous. We need an explicit sensor/camera coordinate visualization mode to compare GT and predictions in the same physical sensor frame.

## What Changes

- Add coordinate-space selection for `visualize_inference_rerun.py` (default existing behavior, plus sensor/camera frame mode).
- Add sensor-frame reference selection for multimodal/multiview samples (at least RGB and depth/lidar paths for HuMMan).
- Transform both GT and predicted 3D keypoints/mesh consistently into the selected sensor frame before rerun logging.
- Set rerun view coordinates to match sensor convention in sensor-frame mode (Y-down camera-style).
- Log coordinate-space and selected sensor-frame metadata under `world/info/*`.
- Keep current workflow/config compatibility for existing commands when the new options are not used.

## Capabilities

### New Capabilities
- `rerun-inference-sensor-frame-visualization`: Sensor/camera coordinate visualization mode for MMHPE inference rerun with explicit reference-frame selection.

### Modified Capabilities
- `rerun-visualization-pipeline`: Extend shared visualization requirements to include coordinate-space controls, sensor-frame metadata, and deterministic frame selection for multiview sensor-frame transforms.

## Impact

- Affected code:
  - `scripts/visualize_inference_rerun.py`
  - `scripts/rerun_utils/session.py`
  - `scripts/rerun_utils/camera.py` (reuse/extend for frame transform utilities)
  - `scripts/rerun_utils/logging3d.py` (if side/front transforms need coordinate-mode branching)
- Affected modalities/datasets:
  - HuMMan V3 first, with RGB + depth/lidar multiview paths in scope
  - mmWave remains out of scope for this first pass
- Config/runtime impact:
  - new CLI/config options for coordinate space and frame reference
  - rerun recordings in `logs/` include explicit coordinate-space metadata
- Non-goals:
  - no model training changes
  - no dataset preprocessing format changes
  - no change to model prediction semantics, only visualization-space mapping
