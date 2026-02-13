## Why

We need a point‑cloud‑only 3D pose estimator to feed FusionFormer without relying on RGB or depth‑derived 2D poses. Mamba4D already exists in the codebase, so aligning it as the HuMMan point‑cloud pose estimator is the fastest low‑risk path.

## What Changes

- Add a training path that uses `models/pc_encoders/mamba4d.py` to predict a **sequence** of 3D keypoints from point clouds.
- Feed the predicted 3D keypoint sequence into FusionFormer as the point‑cloud modality input.
- Add a HuMMan config for point‑cloud‑only training with Mamba4D + FusionFormer.

## Capabilities

### New Capabilities
- `mamba4d-pointcloud-pose-estimator`: Train Mamba4D to output 3D keypoint sequences from point clouds and use them as FusionFormer inputs.

### Modified Capabilities
- 

## Impact

- Affected code: `models/pc_encoders/mamba4d.py` integration, FusionFormer input wiring, HuMMan configs.
- Affected workflows: HuMMan training/eval for point‑cloud‑only pose estimation, FusionFormer fusion with PC‑estimated 3D keypoints.
- Modalities: point cloud only (no RGB/depth usage).
