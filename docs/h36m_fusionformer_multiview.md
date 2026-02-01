# H36M FusionFormer Multiview (RGB)

This configuration uses H36M multi-camera RGB sequences and derives GT 2D keypoints via perspective projection from camera-space 3D joints.

## GT-derived 2D (Perspective Projection)
Given 3D camera-space joints (X, Y, Z) and intrinsics (fx, fy, cx, cy):
- x = fx * (X / Z) + cx
- y = fy * (Y / Z) + cy

## Config
`configs/dev/h36m/h36m_fusionformer_rgb_mv.yml`

## Notes
- Multiview RGB is stacked as [V, T, H, W, 3].
- 17 H36M joints are used (static joints removed).
- seq_len is set to 27 and num_blocks is 2 to match the paper.
