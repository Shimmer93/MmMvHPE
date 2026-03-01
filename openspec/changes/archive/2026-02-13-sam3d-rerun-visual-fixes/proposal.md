## Why

The current SAM-3D-Body rerun output has several correctness issues (RGB color rendering, missing 2D/3D skeleton connectivity, side-view inconsistency, and missing GT overlays), which blocks reliable visual debugging and qualitative comparison in MMHPE benchmark workflows. This should be fixed now so rerun outputs can be used as a trustworthy comparison tool.

## What Changes

- Fix RGB image logging correctness (not only channel order): verify dtype/range/scaling and prevent quantization/clipping that causes near-black images or discrete color bands.
- Add connected 2D keypoint overlay rendering using the correct SAM-3D-Body keypoint topology.
- Align 3D keypoint connectivity and skeleton definition to the SAM-3D-Body keypoint convention (verify upstream definition/order).
- Fix side-view transformation so mesh and 3D keypoints stay consistent in both front and side views.
- Add GT visualization for available sample annotations (at minimum GT 3D keypoints; GT mesh when recoverable from sample fields).
- Keep `scripts/visualize_sam3d_body_rerun.py` CLI compatible, including explicit `--render-mode` behavior.
- Non-goal: no training/evaluation metric changes and no model architecture/training updates.

## Capabilities

### New Capabilities
- `sam3d-rerun-visual-quality`: Correct and consistent SAM-3D-Body rerun rendering for RGB, 2D overlays, and 3D mesh/skeleton views.
- `sam3d-rerun-gt-visualization`: Ground-truth keypoint/mesh visualization in the SAM-3D-Body rerun script when GT is present in the dataset sample.

### Modified Capabilities
- `rerun-visualization-pipeline`: Shared rerun helper behavior is extended to support skeleton connectivity/topology mapping and consistent multi-view transforms for SAM-3D-Body outputs.

## Impact

- Affected code:
  - `scripts/visualize_sam3d_body_rerun.py`
  - `scripts/rerun_utils/logging3d.py`
  - `scripts/rerun_utils/geometry.py`
  - `scripts/rerun_utils/image.py`
  - `docs/rerun_visualization.md`
- Affected modalities/datasets:
  - RGB input visualization (primary)
  - 2D/3D keypoint overlays for HuMMan samples used by SAM-3D-Body rerun flow
- Runtime output impact:
  - Rerun entities under `world/inputs/*`, `world/front/*`, `world/side/*`, and `world/info/*` gain corrected color/edge/topology/GT logs
  - Saved `.rrd` files in `logs/` become suitable for side-by-side qualitative comparison
- Dependencies:
  - No new package dependencies expected; may add static topology mapping data in script/helper code.
