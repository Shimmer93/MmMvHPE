# Panoptic Segmentation Mask Runtime Transforms

## Purpose

`ApplyPanopticSegmentationMask` applies existing SAM3 binary masks to Panoptic RGB and/or depth frames at dataset runtime. It is designed to avoid writing a second masked copy of the dataset to disk.

The transform is Panoptic-specific and expects masks under each sequence root:

```text
<sequence>/sam_segmentation_mask/<camera>/<frame>.png
```

## Implementation

Files:

- `datasets/transforms/panoptic_mask_transforms.py`
- `datasets/panoptic_preprocessed_dataset_v1.py`

The Panoptic dataset now exposes two extra sample fields used by the transform:

- `sequence_root`: absolute path to the sequence directory
- `body_frame_ids`: synchronized body-frame ids for the sample window

The transform:

- accepts `apply_to: ['rgb']`, `['depth']`, or `['rgb', 'depth']`
- resolves mask paths from `sequence_root`, selected camera names, and `body_frame_ids`
- maps normalized sample cameras like `kinect_008` to on-disk folders like `kinect_8`
- zeroes masked-out RGB pixels directly in the RGB image plane
- reprojects the RGB mask into the depth image plane before zeroing depth pixels
- preserves depth dtype / shape while masking
- raises explicit errors on missing masks, unreadable masks, unresolved metadata, or mask/frame shape mismatch

For depth, the transform does not reuse the RGB mask bitmap directly. It:

- back-projects valid depth pixels with `K_depth`
- transforms them from depth camera to RGB camera using the calibrated camera geometry
- projects them with `K_color`
- samples the RGB mask in the RGB image plane
- keeps only depth pixels whose projected RGB samples fall on foreground

## Usage

Insert the transform before normalization / formatting steps in a Panoptic pipeline.

RGB-only example:

- `configs/vis/panoptic_mask_runtime_rgb.yml`

Depth-only example:

- `configs/vis/panoptic_mask_runtime_depth.yml`

Combined RGB+depth example:

- `configs/vis/panoptic_mask_runtime_rgb_depth.yml`

Minimal YAML pattern:

```yaml
- name: ApplyPanopticSegmentationMask
  params:
    apply_to: ['rgb', 'depth']
- name: VideoNormalize
  params:
    norm_mode: 'imagenet'
    keys: ['input_rgb']
- name: ToTensor
  params: null
```

## Important Constraint

Depth masking only applies when `input_depth` is still present in the sample.

If a Panoptic config sets `convert_depth_to_lidar: true`, `PanopticPreprocessedDatasetV1` converts depth to `input_lidar` before the pipeline and removes `input_depth`. In that case this transform cannot modify the depth image because it never reaches the pipeline.

For that reason, the depth example configs in this change use:

```yaml
convert_depth_to_lidar: false
```

If masked depth-derived point clouds are needed later, that should be handled by a separate change.

## Validation Commands

### 1. Syntax check

```bash
uv run python -m py_compile \
  datasets/transforms/panoptic_mask_transforms.py \
  scripts/export_panoptic_mask_validation_samples.py \
  datasets/panoptic_preprocessed_dataset_v1.py
```

### 2. Successful masking validation

This checks that the Panoptic dataset pipeline still instantiates and applies the transform for:

- RGB-only config
- depth-only config
- RGB+depth config

```bash
uv run python - <<'PY'
import yaml
from misc.registry import create_dataset

cfgs = [
    'configs/vis/panoptic_mask_runtime_rgb.yml',
    'configs/vis/panoptic_mask_runtime_depth.yml',
    'configs/vis/panoptic_mask_runtime_rgb_depth.yml',
]
for cfg_path in cfgs:
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    ds, _ = create_dataset(cfg['test_dataset']['name'], dict(cfg['test_dataset']['params']), cfg['test_pipeline'])
    sample = ds[0]
    print('ok', cfg_path, sorted(k for k in sample.keys() if k in {'input_rgb', 'input_depth', 'rgb_camera', 'depth_camera'}))
PY
```

Observed output:

```text
ok configs/vis/panoptic_mask_runtime_rgb.yml ['input_rgb', 'rgb_camera']
ok configs/vis/panoptic_mask_runtime_depth.yml ['depth_camera', 'input_depth']
ok configs/vis/panoptic_mask_runtime_rgb_depth.yml ['depth_camera', 'input_depth', 'input_rgb', 'rgb_camera']
```

### 3. Failure-case validation

This checks RGB-side fail-fast behavior for:

- missing mask file
- unreadable mask file
- missing frame metadata
- mask/frame shape mismatch

```bash
uv run python - <<'PY'
import tempfile
from pathlib import Path
import cv2
import numpy as np
from datasets.transforms.panoptic_mask_transforms import ApplyPanopticSegmentationMask

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / 'seq1'
    (root / 'sam_segmentation_mask' / 'kinect_8').mkdir(parents=True)
    rgb = np.full((4, 5, 3), 7, dtype=np.uint8)
    ok_mask = np.zeros((4, 5), dtype=np.uint8)
    ok_mask[1:3, 2:5] = 255
    cv2.imwrite(str(root / 'sam_segmentation_mask' / 'kinect_8' / '00000001.png'), ok_mask)
    t = ApplyPanopticSegmentationMask(apply_to=['rgb'])
    sample = {
        'sequence_root': str(root),
        'selected_cameras': {'rgb': ['kinect_008'], 'depth': [], 'lidar': []},
        'body_frame_ids': [1],
        'input_rgb': [rgb],
    }
    out = t(sample)
    assert int((out['input_rgb'][0] != 0).any(axis=2).sum()) == int((ok_mask > 0).sum())

    try:
        t({'sequence_root': str(root), 'selected_cameras': {'rgb': ['kinect_008'], 'depth': [], 'lidar': []}, 'body_frame_ids': [2], 'input_rgb': [rgb]})
        raise AssertionError('expected missing-mask failure')
    except FileNotFoundError:
        pass

    bad_path = root / 'sam_segmentation_mask' / 'kinect_8' / '00000003.png'
    bad_path.write_bytes(b'not_an_image')
    try:
        t({'sequence_root': str(root), 'selected_cameras': {'rgb': ['kinect_008'], 'depth': [], 'lidar': []}, 'body_frame_ids': [3], 'input_rgb': [rgb]})
        raise AssertionError('expected unreadable-mask failure')
    except RuntimeError:
        pass

    wrong_mask = np.zeros((3, 5), dtype=np.uint8)
    cv2.imwrite(str(root / 'sam_segmentation_mask' / 'kinect_8' / '00000004.png'), wrong_mask)
    try:
        t({'sequence_root': str(root), 'selected_cameras': {'rgb': ['kinect_008'], 'depth': [], 'lidar': []}, 'body_frame_ids': [4], 'input_rgb': [rgb]})
        raise AssertionError('expected shape-mismatch failure')
    except ValueError:
        pass

    try:
        t({'sequence_root': str(root), 'selected_cameras': {'rgb': ['kinect_008'], 'depth': [], 'lidar': []}, 'input_rgb': [rgb]})
        raise AssertionError('expected unresolved-metadata failure')
    except ValueError:
        pass

print('ok failure validation')
PY
```

Observed output:

```text
ok failure validation
```

### 4. Manual visual inspection export

This script exports:

- original RGB image
- RGB image with mask overlay
- masked RGB image
- original depth image and colorized preview
- naive depth-masked image and colorized preview
- reprojected depth-masked image and colorized preview
- masked and unmasked point clouds from depth as `.ply` and `.npy`

```bash
uv run python scripts/export_panoptic_mask_validation_samples.py \
  --sequence 170915_office1 \
  --camera kinect_8 \
  --frame-ids 00004455 \
  --out-dir logs/panoptic_mask_validation_reproject_v2
```

Observed output:

```text
[panoptic-mask-validation] wrote: logs/panoptic_mask_validation_reproject_v2/170915_office1/kinect_8/depth
[panoptic-mask-validation] summary: logs/panoptic_mask_validation_reproject_v2/170915_office1/kinect_8/depth/export_summary.json
```

Example exported files:

- `logs/panoptic_mask_validation_reproject_v2/170915_office1/kinect_8/depth/00004455/depth_masked_naive.png`
- `logs/panoptic_mask_validation_reproject_v2/170915_office1/kinect_8/depth/00004455/depth_masked.png`
- `logs/panoptic_mask_validation_reproject_v2/170915_office1/kinect_8/depth/00004455/pointcloud_masked_naive.ply`
- `logs/panoptic_mask_validation_reproject_v2/170915_office1/kinect_8/depth/00004455/pointcloud_masked.ply`

Observed output:

```text
[panoptic-mask-validation] wrote: logs/panoptic_mask_validation/170915_office1/kinect_8
[panoptic-mask-validation] summary: logs/panoptic_mask_validation/170915_office1/kinect_8/export_summary.json
```

Export root:

- `logs/panoptic_mask_validation/170915_office1/kinect_8`

Example files:

- `logs/panoptic_mask_validation/170915_office1/kinect_8/00004455/rgb_mask_overlay.jpg`
- `logs/panoptic_mask_validation/170915_office1/kinect_8/00004455/depth_masked_preview_jet.jpg`
- `logs/panoptic_mask_validation/170915_office1/kinect_8/00004455/pointcloud_unmasked.ply`
- `logs/panoptic_mask_validation/170915_office1/kinect_8/00004455/pointcloud_masked.ply`
