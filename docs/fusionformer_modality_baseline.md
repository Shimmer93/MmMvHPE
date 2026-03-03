# FusionFormer Modality Baseline

This baseline adapts FusionFormer (AAAI-24) to HummanPreprocessedDatasetV2 by treating modalities (RGB and depth/PC) as “views.” It predicts 3D keypoints only and ignores camera intrinsics during fusion.

## Assumptions
- Modalities represent different viewpoints of the same subject; fusion is camera-parameter-free.
- GT 2D/3D poses are used as stand-ins for off-the-shelf estimators.
- Output is 24-joint 3D keypoints (SMPL joint order used by the dataset).

## Quick Run (mini)
```bash
uv run python main.py -c configs/dev/humman_fusionformer_gt_pose_mini.yml
```

## Config Notes
- `rgb_input_key` / `depth_input_key` point to GT pose inputs created by `AttachGTPoseInputs`.
- `return_keypoints_sequence` enables sequence GT keypoints for temporal fusion.
