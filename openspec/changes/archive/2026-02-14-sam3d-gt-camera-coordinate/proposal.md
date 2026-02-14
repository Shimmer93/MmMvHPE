## Why

SAM-3D-Body predictions are in camera coordinates, while HuMMan V3 GT keypoints are currently visualized in canonical pelvis-centered space. This coordinate mismatch makes side-by-side qualitative checks misleading and blocks reliable benchmark comparison.

## What Changes

- Add an explicit GT coordinate-space option for SAM-3D-Body rerun visualization (`canonical` vs `camera`).
- Use dataset camera extrinsics to transform HuMMan V3 GT 3D keypoints from canonical space to per-view camera space when requested.
- Apply the same coordinate-space selection to temporal GT sequences so multiframe visualization stays frame-consistent.
- Add clear logging/metadata in rerun output so each recording states which GT space was used.
- Add config and CLI controls for coordinate-space selection, with deterministic behavior in single-view and multi-view runs.

## Capabilities

### New Capabilities

- `sam3d-rerun-coordinate-alignment`: Coordinate-space alignment controls and transforms for SAM-3D-Body vs HuMMan V3 GT visualization.

### Modified Capabilities

- `sam3d-rerun-gt-visualization`: Extend GT visualization requirements to support canonical/camera-space GT selection and temporal consistency.
- `sam3d-rerun-multiframe-visualization`: Ensure multiframe GT visualization applies the same selected coordinate space per frame and per view.

## Impact

- Affected code:
  - `scripts/visualize_sam3d_body_rerun.py`
  - helper modules under `scripts/` used for camera extraction, projection, and rerun logging
  - HuMMan V3 sample parsing paths used to access `rgb_camera` / selected camera metadata
- Affected dataset/modality scope:
  - HuMMan V3 first (RGB multiview visualization; depth/LiDAR/mmWave unchanged in this change)
- Config/runtime impact:
  - Add a new visualization config option for GT coordinate space
  - Rerun outputs under `logs/` will include explicit coordinate-space metadata
- Non-goals:
  - No model training/inference algorithm changes
  - No redefinition of dataset storage format
  - No change to SAM-3D-Body predicted output coordinate frame
