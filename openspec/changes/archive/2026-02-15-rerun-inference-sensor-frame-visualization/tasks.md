## 1. Coordinate-mode CLI and metadata wiring

- [x] 1.1 Add CLI options in `scripts/visualize_inference_rerun.py` for coordinate mode (`world|sensor`), reference sensor label, and reference view index.
- [x] 1.2 Enforce explicit reference selection when `coord-space=sensor` and fail fast with actionable errors on missing/invalid values.
- [x] 1.3 Log coordinate metadata (`coord_space`, `reference_sensor`, `reference_view`) under `world/info/*` for every sample.

## 2. Sensor-frame transform integration

- [x] 2.1 Reuse/extend `scripts/rerun_utils/camera.py` to resolve selected sensor extrinsics and support `lidar` label mapping for depth-derived point cloud camera metadata.
- [x] 2.2 Implement sensor-frame transform flow for GT and predicted 3D keypoints in `scripts/visualize_inference_rerun.py` and ensure shape-safe handling for temporal data.
- [x] 2.3 Integrate mesh vertex transforms in sensor mode so GT/pred mesh and keypoints are expressed in the same selected sensor frame.
- [x] 2.4 Keep world-mode path unchanged and isolate sensor-mode branch to avoid mixing world flips/rotations with sensor transforms.

## 3. Rerun coordinate declaration and view behavior

- [x] 3.1 Set rerun view coordinates to `RIGHT_HAND_Y_DOWN` in sensor mode and keep `RIGHT_HAND_Y_UP` in world mode via `scripts/rerun_utils/session.py`.
- [x] 3.2 Ensure front/side skeleton and mesh logging use consistent side-transform behavior per mode (no mirrored mismatch between GT and prediction).
- [x] 3.3 Validate multiview determinism: selected reference sensor/view is applied consistently across logged frames for a sample.

## 4. Validation and documentation

- [x] 4.1 Run smoke checks in world mode and sensor mode using `uv run --no-sync python scripts/visualize_inference_rerun.py ...` with a HuMMan demo config and save `.rrd` outputs under `logs/rerun_smoke/`.
- [ ] 4.2 Verify sensor-mode `.rrd` includes `world/info/*` coordinate metadata and visually confirm GT/pred alignment in the selected sensor frame.
- [x] 4.3 Update `docs/rerun_visualization.md` with new CLI usage, sensor-mode requirements (explicit reference selection), and examples for RGB and `lidar` reference labels.
