## 1. Dataset and 2D Adapter

- [x] 1.1 Add an H36M multiview dataset class that returns RGB as [V, T, H, W, 3] and per-frame GT 3D joints
- [x] 1.2 Add GT-derived 2D keypoint adapter using perspective projection (fx, fy, cx, cy)
- [x] 1.3 Ensure camera list is configurable via config and preserved in sample metadata

## 2. Model Wiring and Config

- [x] 2.1 Add FusionFormer H36M config with seq_len=27, num_blocks=2, num_joints=17, all cameras
- [x] 2.2 Wire GT 2D adapter outputs into model inputs (Pose2DBackbone + FusionFormer aggregator)

## 3. Validation

- [x] 3.1 Run a short `uv run` sanity check on the H36M config to verify multiview data + forward pass
- [x] 3.2 Document how GT 2D is derived and any limitations in a short note (if needed under docs/)
