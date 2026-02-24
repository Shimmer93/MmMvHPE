# Single-Model LiDAR HPE

This module provides a standalone LiDAR-only human pose estimation pipeline:
- dataset: depth-to-point-cloud HuMMan loader with root-centered new-world keypoints,
- model: `MAMBA4DEncoder` + MLP keypoint regressor,
- runner: argparse-only training/testing entrypoint (`tools/single_model_hpe/main_lidar.py`).

Training uses the following transform chain (in this order):
- `CameraParamToPoseEncoding(pose_encoding_type="absT_quaR_FoV")`
- `PCCenterWithKeypoints(center_type="mean", keys=["input_lidar"], keypoints_key="gt_keypoints")`
- `PCPad(num_points=1024, pad_mode="repeat", keys=["input_lidar"])`
- `ToTensor()`

## Files

- `tools/single_model_hpe/dataset.py`: HuMMan depth-only dataset that outputs:
  - `input_lidar`: `(T, N, 3)` depth-derived point cloud sequence,
  - `gt_keypoints`: `(24, 3)` 3D skeleton in the same root-centered coordinate system.
- `tools/single_model_hpe/model.py`: simple LiDAR HPE model.
- `tools/single_model_hpe/train_eval.py`: train/eval/test loops and checkpoint helpers.
- `tools/single_model_hpe/main_lidar.py`: command-line entrypoint.

## Data assumptions

`--data-root` should follow the preprocessed HuMMan structure used by `datasets/humman_dataset_v2.py`:
- `depth/`
- `cameras/`
- `smpl/`
- `skl/`

Depth images are projected to point clouds via camera intrinsics/extrinsics, then transformed to the same root-centric frame as the 3D skeleton target.

## Examples

Train and test:

```bash
uv run python tools/single_model_hpe/main_lidar.py \
  --data-root /opt/data/humman_cropped \
  --epochs 20 \
  --batch-size 8 \
  --num-points 1024 \
  --output-dir logs/single_model_hpe/run1
```

Test only from checkpoint:

```bash
uv run python tools/single_model_hpe/main_lidar.py \
  --data-root /opt/data/humman_cropped \
  --test-only \
  --checkpoint logs/single_model_hpe/run1/best.pt \
  --output-dir logs/single_model_hpe/run1
```

Export frame-wise predictions for the entire depth dataset as MMPose-style JSON:

```bash
uv run python tools/single_model_hpe/main_lidar.py \
  --data-root /opt/data/humman_cropped \
  --test-only \
  --checkpoint logs/single_model_hpe/run1/best.pt \
  --output-dir logs/single_model_hpe/run1 \
  --export-all-json all_frame_predictions.json
```

The JSON payload follows `tools/predict_2d_skeletons_mmpose.py` top-level schema:
- `input_dir`, `config`, `checkpoint`, `device`, `num_images`, `predictions`.
