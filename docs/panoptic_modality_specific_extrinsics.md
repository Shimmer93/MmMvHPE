# Panoptic Modality-Specific Extrinsics

## Purpose

Correct the Panoptic runtime camera geometry so:

- RGB uses color-camera extrinsics
- depth uses depth-camera extrinsics
- depth-derived LiDAR follows depth-camera extrinsics

This change fixes a dataset-layer geometry issue in `PanopticPreprocessedDatasetV1`. It does not solve RGB-mask-to-depth reprojection or silhouette differences between sensors.

## Problem

The preprocessed Panoptic metadata includes:

- `M_color`
- `M_depth`
- `extrinsic_world_to_color`

Before this change, `datasets/panoptic_preprocessed_dataset_v1.py` used `extrinsic_world_to_color` for all modalities whenever that field existed in `meta/cameras_kinect_cropped.json`.

On this machine, that field is present in all preprocessed Panoptic sequences, so depth and depth-derived LiDAR were sharing RGB/world-to-color extrinsics at runtime even though the metadata contains distinct depth geometry.

## Runtime Contract

File:

- `datasets/panoptic_preprocessed_dataset_v1.py`

Current extrinsic precedence:

- `rgb`:
  - prefer `extrinsic_world_to_color` when present and valid
  - otherwise use Panoptic calibration fallback when available
- `depth`:
  - compute `world_to_depth = inv(M_depth) @ M_color @ world_to_color`
  - do not fall back to `extrinsic_world_to_color` directly
  - do not use `M_world2sensor` as the Panoptic world frame
- `lidar` (Panoptic depth-derived LiDAR):
  - use the same extrinsic as `depth`

If required color-world or depth-relative geometry is missing or malformed, dataset loading now fails explicitly instead of silently reusing RGB extrinsics.

## Scope Boundary

This change is only about correcting runtime extrinsics.

Out of scope:

- reprojecting RGB masks into the depth image plane
- rectifying RGB/depth silhouettes
- rewriting all preprocessed metadata files on disk

Correct extrinsics are a prerequisite for later reprojection work, but they are not the same feature.

## Compatibility

No changes are required in:

- `main.py`
- `datasets/data_api.py`
- model APIs
- YAML config keys

But this is a breaking geometry change for Panoptic depth/LiDAR sensor-frame behavior. Any training, evaluation, or visualization that depends on depth or depth-derived LiDAR camera extrinsics may shift.

I checked the main consumers that read `depth_camera` / `lidar_camera` metadata:

- `scripts/rerun_utils/camera.py`
- `tools/eval_fixed_lidar_frame.py`
- `scripts/export_panoptic_pipeline_debug_sample.py`

They consume camera payloads generically and did not require code changes for this fix.

## Validation Commands

### 1. Syntax check

```bash
uv run python -m py_compile \
  datasets/panoptic_preprocessed_dataset_v1.py \
  scripts/export_panoptic_modality_extrinsics_debug.py
```

### 2. Numeric validation across representative sequences

This verifies that RGB and depth-derived LiDAR extrinsics are no longer identical for the same Kinect node, but now differ only by the small Kinect color-depth baseline.

```bash
uv run python - <<'PY'
import yaml, numpy as np
from misc.registry import create_dataset

checks = [
    ('170915_office1', 'kinect_8'),
    ('170407_office2', 'kinect_9'),
    ('171026_cello3', 'kinect_8'),
]
with open('configs/exp/panoptic/cross_camera_split/hpe.yml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

for seq, cam in checks:
    params = dict(cfg['val_dataset']['params'])
    params['split_config'] = None
    params['sequence_allowlist'] = [seq]
    params['rgb_cameras'] = [cam]
    params['depth_cameras'] = [cam]
    params['rgb_cameras_per_sample'] = 1
    params['depth_cameras_per_sample'] = 1
    params['lidar_cameras_per_sample'] = 1
    params['convert_depth_to_lidar'] = True
    ds, _ = create_dataset(cfg['val_dataset']['name'], params, [])
    s = ds[0]
    rgb = np.asarray(s['rgb_camera']['extrinsic'], dtype=np.float32)
    lidar = np.asarray(s['lidar_camera']['extrinsic'], dtype=np.float32)
    print(seq, cam, 'allclose=', bool(np.allclose(rgb, lidar)), 'trans_delta=', (lidar[:, 3] - rgb[:, 3]).tolist())
    if np.allclose(rgb, lidar):
        raise SystemExit(f'Extrinsics still identical for {seq} {cam}')
PY
```

Observed output:

```text
170915_office1 kinect_8 allclose= False angle_deg= 0.3238512873649597 t_delta= [0.059984803199768066, 0.008559823036193848, -0.0009102821350097656] t_norm= 0.06059930473566055
170407_office2 kinect_9 allclose= False angle_deg= 0.37949177622795105 t_delta= [0.06241416931152344, 0.017042160034179688, 0.008471012115478516] t_norm= 0.06525122374296188
171026_cello3 kinect_8 allclose= False angle_deg= 0.3238512873649597 t_delta= [0.06002640724182129, 0.007843613624572754, -0.0008449554443359375] t_norm= 0.06054259464144707
```

### 3. Debug export for one sample

This writes:

- sensor-frame LiDAR points
- world-frame LiDAR points using corrected depth extrinsic
- world-frame LiDAR points using the legacy RGB extrinsic for comparison
- metadata JSON with RGB and effective depth-based extrinsics

```bash
uv run python scripts/export_panoptic_modality_extrinsics_debug.py \
  --config configs/exp/panoptic/cross_camera_split/hpe.yml \
  --split val \
  --sequence 170915_office1 \
  --camera kinect_8 \
  --index 0 \
  --out-dir logs/panoptic_modality_extrinsics_debug
```

Observed output:

```text
[panoptic-extrinsics-debug] wrote: logs/panoptic_modality_extrinsics_debug_v2/170915_office1/00000153_kinect_008
[panoptic-extrinsics-debug] meta: logs/panoptic_modality_extrinsics_debug_v2/170915_office1/00000153_kinect_008/meta.json
```

Exported files:

- `logs/panoptic_modality_extrinsics_debug_v2/170915_office1/00000153_kinect_008/meta.json`
- `logs/panoptic_modality_extrinsics_debug_v2/170915_office1/00000153_kinect_008/lidar_sensor_points.ply`
- `logs/panoptic_modality_extrinsics_debug_v2/170915_office1/00000153_kinect_008/lidar_world_from_depth_extrinsic.ply`
- `logs/panoptic_modality_extrinsics_debug_v2/170915_office1/00000153_kinect_008/lidar_world_from_rgb_extrinsic_legacy_compare.ply`
