# Tools and Scripts

This page summarizes helper scripts in `tools/` and gives runnable command examples.

All examples use `uv run`.

## Dataset preprocessing

### `tools/data_preprocess.py`

Purpose:
- preprocess MMFi/HuMMan raw data into flattened modality folders,
- optionally perform HuMMan cropping with YOLO.

Current status:
- no argparse interface; configured in the `if __name__ == "__main__"` block.
- typical usage is editing the bottom block and running:

```bash
uv run python tools/data_preprocess.py
```

For HuMMan data layout details, see `datasets/HUMMAN_PREPROCESSING.md`.

## Detection helpers

### `tools/image_detection.py`

Purpose:
- run YOLO person detection on RGB images and write JSON annotations.

Example:

```bash
uv run python tools/image_detection.py \
  --input "data/mmfi/rgb/*.jpg" \
  --output data/mmfi/rgb_boxes.json \
  --weights yolov8n.pt --conf 0.25 --iou 0.45
```

### `tools/depth_yolo_detect.py`

Purpose:
- detect the best human in one depth image and save visualization image.

Example:

```bash
uv run python tools/depth_yolo_detect.py \
  data/sample_depth.png \
  logs/depth_detect_vis.png \
  --weights yolov8n.pt --conf 0.25
```

### `tools/pc_detection.py`

Purpose:
- run OpenPCDet detector on `.npy` point clouds and write JSON detections.

Example:

```bash
uv run python tools/pc_detection.py \
  --config weights/pc_detection/pointpillar.yaml \
  --checkpoint weights/pc_detection/pointpillar_7728.pth \
  --input "data/mmfi/lidar/*.npy" \
  --output data/mmfi/lidar_boxes.json \
  --class-name Pedestrian
```

## Pose/skeleton generation

### `tools/predict_2d_skeletons_mmpose.py`

Purpose:
- run MMPose top-down inference on a directory and export 2D keypoints JSON.

Example:

```bash
uv run python tools/predict_2d_skeletons_mmpose.py \
  data/images \
  third_party/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
  <mmpose_checkpoint.pth> \
  data/keypoints_2d.json \
  --device cuda:0
```

### `tools/generate_humman_smpl_outputs.py`

Purpose:
- convert HuMMan SMPL params into saved 3D keypoints/vertices (`keypoints_3d.npz`).

Example:

```bash
uv run python tools/generate_humman_smpl_outputs.py \
  --data_root /data/shared/humman_release_v1.0_point \
  --smpl_model_path weights/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl \
  --device cuda
```

## Camera and geometric utilities

### `tools/test_video_normalize_fov.py`

Purpose:
- test `VideoNormalizeFoV` warp with provided camera intrinsics.

Example:

```bash
uv run python tools/test_video_normalize_fov.py \
  data/input.jpg logs/warped.jpg \
  --fx 1200 --fy 1200 --cx 960 --cy 540 \
  --target-fov 1.0 1.0 --save-side-by-side
```

### `tools/eval_fixed_lidar_frame.py`

Purpose:
- evaluate MPJPE/PA-MPJPE after projecting canonical keypoints into a fixed sensor frame.
- supports two projection modes:
  - `seq_lidar_ref`: legacy fixed per-sequence LiDAR reference camera.
  - `multi_sensor`: fuse fixed per-sequence cameras from multiple modalities (for example `rgb,lidar`) into a target frame.
- robust options for `multi_sensor`:
  - `--fusion-mode weighted|hard_gate`
  - `--reliability-source cross_sensor|temporal|hybrid`

Examples:

```bash
uv run python tools/eval_fixed_lidar_frame.py \
  --pred-file logs/dev_humman/run_x/model_test_predictions.pkl \
  --projection-mode seq_lidar_ref
```

```bash
uv run python tools/eval_fixed_lidar_frame.py \
  --pred-file logs/dev_humman/run_x/model_test_predictions.pkl \
  --projection-mode multi_sensor \
  --target-modality lidar \
  --fusion-modalities rgb,lidar
```

```bash
uv run python tools/eval_fixed_lidar_frame.py \
  --pred-file logs/dev_humman/run_x/model_test_predictions.pkl \
  --projection-mode multi_sensor \
  --target-modality lidar \
  --fusion-modalities rgb,lidar \
  --fusion-mode hard_gate \
  --reliability-source hybrid
```

### `tools/fix_pred_to_static_sensors.py`

Purpose:
- post-process predicted cameras/keypoints to a static multi-sensor frame.

Example:

```bash
uv run python tools/fix_pred_to_static_sensors.py \
  logs/dev_humman/run_x/model_test_predictions.pkl \
  logs/dev_humman/run_x/model_test_predictions_fixed.pkl \
  --modalities rgb depth lidar \
  --iters 500 --lr 1e-2 --overwrite-pred-keypoints
```

## SMPL fitting and visualization

### `tools/optimize_smpl_for_skl.py`

Purpose:
- fit SMPL parameters to predicted/GT keypoints in prediction pickles.

Example:

```bash
uv run python tools/optimize_smpl_for_skl.py \
  --preds_path logs/dev_humman/run_x/model_test_predictions.pkl \
  --model_folder weights \
  --gender neutral \
  --batch_size 100 \
  --num_iters 2000 \
  --lr 0.02
```

### `tools/vis_smpl.py`

Purpose:
- visualize SMPL and keypoints with rerun.

Current status:
- no argparse interface; edit paths in the main block and run:

```bash
uv run python tools/vis_smpl.py
```

## Single-model lidar HPE

### `tools/single_model_hpe/main_lidar.py`

Purpose:
- run a standalone LiDAR-only HPE pipeline (no config file dependency),
- train on depth-derived point clouds and output test predictions.

Example:

```bash
uv run python tools/single_model_hpe/main_lidar.py \
  --data-root /opt/data/humman_cropped \
  --epochs 20 \
  --batch-size 8 \
  --num-points 1024 \
  --output-dir logs/single_model_hpe/run1
```

## Script writing guideline

When adding new scripts:
- add argparse and `--help` text where possible,
- keep default paths repo-relative,
- add at least one example command in this file.
