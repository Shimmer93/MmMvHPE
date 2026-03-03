## 1. Coordinate-space interface and plumbing

- [x] 1.1 Add `gt_coordinate_space` option (`canonical`/`camera`) to `configs/demo/humman_sam3d_body_vis.yml` with default `canonical`.
- [x] 1.2 Add CLI override support in `scripts/visualize_sam3d_body_rerun.py` and validate allowed values (`canonical`, `camera`) with a clear error on invalid input.
- [x] 1.3 Ensure rerun metadata under `world/info/*` logs `gt_coordinate_space` for every run.

## 2. GT canonical->camera conversion

- [x] 2.1 Add a helper in `scripts` visualization utilities to extract per-view extrinsic matrices from `sample["rgb_camera"]` and normalize single-view/multiview handling.
- [x] 2.2 Implement GT transform `X_cam = R * X + t` for 3D keypoints with shape-preserving behavior for `(..., J, 3)`.
- [x] 2.3 Integrate conversion into `scripts/visualize_sam3d_body_rerun.py` so GT is transformed only when `gt_coordinate_space=camera`.
- [x] 2.4 Add fail-fast checks in camera mode when required `rgb_camera` metadata is missing or mismatched with selected views.

## 3. Multiframe consistency

- [x] 3.1 Apply the same coordinate-space mode to temporal GT (`gt_keypoints_seq`) for all selected frames.
- [x] 3.2 Ensure per-frame/per-view camera-space transforms are used in multiframe camera mode and that canonical mode bypasses transforms for all frames.
- [x] 3.3 Keep single-mode behavior (no mixed canonical+camera GT panels in one run) and assert this in script control flow.

## 4. Validation and documentation

- [x] 4.1 Run smoke tests with `uv run` for both modes and multiframe: `uv run python scripts/visualize_sam3d_body_rerun.py --config configs/demo/humman_sam3d_body_vis.yml --render-mode mesh --num-frames 3 --gt-coordinate-space canonical` and `uv run python scripts/visualize_sam3d_body_rerun.py --config configs/demo/humman_sam3d_body_vis.yml --render-mode mesh --num-frames 3 --gt-coordinate-space camera`.
- [x] 4.2 Verify generated `.rrd` metadata includes `gt_coordinate_space` and visually confirm side-view GT/prediction alignment improves in camera mode.
- [x] 4.3 Add/update docs in `docs/` describing coordinate-space semantics, required camera metadata, and example commands for both modes.
