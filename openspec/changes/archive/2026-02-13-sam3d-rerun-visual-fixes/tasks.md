## 1. Image Integrity and Overlay Pipeline

- [x] 1.1 Trace image flow in `scripts/visualize_sam3d_body_rerun.py` from sample loading to `rr.Image` and add explicit shape/dtype/range checks.
- [x] 1.2 Fix RGB handling so rerun receives correct image values (prevent near-black/discrete-color collapse), including any required cv2 conversion boundaries.
- [x] 1.3 Add optional debug logging for image min/max/dtype before rerun logging and validate on one sample.

## 2. SAM-3D-Body Keypoint Topology Alignment

- [x] 2.1 Inspect `third_party/sam-3d-body` to identify keypoint order and skeleton connectivity source for predicted 2D/3D keypoints.
- [x] 2.2 Implement SAM-3D-Body skeleton edge mapping in visualization code (`scripts/visualize_sam3d_body_rerun.py` and/or `scripts/rerun_utils/logging3d.py`).
- [x] 2.3 Update 2D overlay rendering to draw connected skeleton edges in addition to keypoint markers.
- [x] 2.4 Update 3D skeleton logging to use the SAM-3D-Body edge mapping (not default MMHPE skeleton edges).

## 3. 3D Transform and View Consistency

- [x] 3.1 Refactor mesh/keypoint coordinate transforms in `scripts/visualize_sam3d_body_rerun.py` to use one shared front-space transform.
- [x] 3.2 Derive side-view mesh and side-view keypoints from the same transformed coordinates and verify visual alignment in rerun.
- [x] 3.3 Ensure behavior is consistent across `--render-mode mesh` and `--render-mode auto` when mesh is available.

## 4. Ground-Truth Visualization

- [x] 4.1 Add GT keypoint logging path in `scripts/visualize_sam3d_body_rerun.py` using sample fields when present.
- [x] 4.2 Add GT mesh logging path when sufficient GT SMPL fields are available, reusing existing SMPL conversion helpers where applicable.
- [x] 4.3 Add `world/info/*` metadata flags for GT availability (keypoints/mesh present vs missing) without failing inference visualization.

## 5. Validation and Documentation

- [x] 5.1 Run SAM-3D-Body rerun smoke tests with `overlay`, `auto`, and `mesh` modes using `uv run` commands and save `.rrd` outputs under `logs/`.
- [x] 5.2 Validate that input RGB appearance, 2D skeleton connectivity, 3D skeleton connectivity, and side-view alignment are corrected in rerun.
- [x] 5.3 Update `docs/rerun_visualization.md` with: topology source path in SAM repo, image format contract, render-mode behavior, GT logging behavior, and troubleshooting notes.
