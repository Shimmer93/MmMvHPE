# Panoptic Best-Depth Camera Search

`scripts/find_best_panoptic_depth_camera_sensor.py` ranks Panoptic `rgb/depth` camera pairs from a saved `*_test_predictions.pkl` using the same per-frame sensor-space projection logic as `tools/eval_per_frame_sensor.py`.

It is intended for cases where the model was already evaluated on many camera pairs and you want to choose the best depth/LiDAR partner for a specific RGB camera without rerunning inference.

## What it measures

- `sensor_mpjpe_m`: MPJPE after projecting prediction and GT into the target sensor frame
- `sensor_pampjpe_m`: PA-MPJPE in the same sensor frame
- `sensor_centered_mpjpe_m`: pelvis-centered MPJPE in the same sensor frame

The target sensor defaults to LiDAR sensor index `0`, which matches the current Panoptic RGB+LiDAR evaluation path.

## Example

```bash
uv run python scripts/find_best_panoptic_depth_camera_sensor.py \
  --pred-file logs/occlusion_robustness_seq1_runtime_mask_rgb_lidar_eval/20260316_rerun_final_eval_occluded_sensor/PanopticHPE_test_predictions.pkl \
  --sequence 170407_office2 \
  --rgb-cameras kinect_007,kinect_009 \
  --workers 8 \
  --out-json logs/panoptic_depth_camera_search/office2_rgb7_rgb9_sensor.json \
  --out-csv logs/panoptic_depth_camera_search/office2_rgb7_rgb9_sensor.csv
```

## Output

The script prints ranked rows to stdout and can also save JSON / CSV.

Each row includes:

- `sequence`
- `rgb_camera`
- `depth_camera`
- `sample_count`
- `frame_start`
- `frame_end`
- `sensor_mpjpe_m`
- `sensor_pampjpe_m`
- `sensor_centered_mpjpe_m`
