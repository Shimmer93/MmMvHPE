## Why

We need a dedicated visualization script for the preprocessed H36M multiview dataset to
verify that 3D keypoints project correctly into 2D image space. The current visualization
does not overlay multiview 2D keypoints on RGB images, which makes projection errors hard
to diagnose.

## What Changes

- Add a new script under `scripts/` to visualize preprocessed H36M data.
- The script loads a specified training config and split, then overlays N-view 2D keypoints
  on the corresponding RGB images and renders the GT 3D skeleton for the same sample.
- Provide CLI flags for config path, split, and sample index/number.

## Capabilities

### New Capabilities
- `h36m-preprocessed-visualizer`: Visualize multiview 2D keypoints over RGB images and GT 3D skeleton
  for preprocessed H36M using a specified config/split/sample.

### Modified Capabilities
- 

## Impact

- Affected code: new script under `scripts/`, minor use of existing dataset and visualization utilities.
- Affected workflows: dataset inspection and debugging of 3D→2D projection for H36M multiview.
- Dependencies/APIs: no new dependencies; uses existing `uv run` execution flow.
