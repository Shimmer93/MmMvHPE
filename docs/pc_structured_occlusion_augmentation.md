# PC Structured Occlusion Augmentation

This document describes the runtime LiDAR augmentation added by `PCStructuredOcclusionAug` in `datasets/transforms/pc_transforms.py`.

The transform is intended for training-time use only. It removes contiguous regions from `input_lidar` in a range-image-style view, which is useful for simulating occlusion from other objects or structured sensor dropout without regenerating dataset artifacts.

## Behavior

- projects each LiDAR frame into a 2D angular view using azimuth and elevation
- samples one or more blobs in that range-image space
- removes the corresponding 3D points from `input_lidar`
- preserves a configurable minimum number of points so downstream centering and padding remain valid

This augmentation should run before `PCCenterWithKeypoints`.

## Recommended Conservative Defaults

These settings are intended as a safe starting point for synthetic pretraining or synthetic-to-real finetuning:

```yaml
- name: PCStructuredOcclusionAug
  params:
    apply_prob: 0.35
    range_image_size: [64, 256]
    blob_count_range: [1, 2]
    blob_shape_mode: mixed
    rectangle_height_ratio_range: [0.08, 0.18]
    rectangle_width_ratio_range: [0.08, 0.18]
    circle_radius_ratio_range: [0.06, 0.12]
    min_points_kept: 256
    keys: ['input_lidar']
```

Useful ablation knobs:
- `apply_prob`: overall frequency of occlusion augmentation
- `blob_count_range`: how many missing regions to create
- `blob_shape_mode`: `rectangle`, `circle`, or `mixed`
- `rectangle_height_ratio_range` and `rectangle_width_ratio_range`: angular size of rectangular holes
- `circle_radius_ratio_range`: angular size of circular holes
- `min_points_kept`: lower bound that protects the downstream centering and padding path

## Example Synthetic-Transfer Pipeline

Insert the augmentation before `PCCenterWithKeypoints`:

```yaml
train_pipeline:
  - name: VideoCenterCropResize
    params:
      size: [224, 224]
      keys: ['input_rgb']
  - name: CameraParamToPoseEncoding
    params:
      pose_encoding_type: "absT_quaR_FoV"
  - name: PCStructuredOcclusionAug
    params:
      apply_prob: 0.35
      range_image_size: [64, 256]
      blob_count_range: [1, 2]
      blob_shape_mode: mixed
      rectangle_height_ratio_range: [0.08, 0.18]
      rectangle_width_ratio_range: [0.08, 0.18]
      circle_radius_ratio_range: [0.06, 0.12]
      min_points_kept: 256
      keys: ['input_lidar']
  - name: PCCenterWithKeypoints
    params:
      center_type: mean
      keys: ['input_lidar']
      keypoints_key: 'gt_keypoints'
  - name: VideoNormalize
    params:
      norm_mode: imagenet
      keys: ['input_rgb']
  - name: PCPad
    params:
      num_points: 1024
      pad_mode: repeat
      keys: ['input_lidar']
  - name: ToTensor
    params: null
```

The same transform block can be used in HuMMan-style synthetic pretraining, Panoptic-style synthetic pretraining, or real-data finetuning configs that already use `input_lidar`.

## Validation Commands

Loader-level validation with synthetic HuMMan export:

```bash
uv run python - <<'PY'
from copy import deepcopy
from datasets import *
from misc.registry import create_dataset
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/exp/synthetic_transfer/humman/synthetic_stage1_pretrain.yml")
pipeline = deepcopy(cfg.train_pipeline)
pipeline.insert(2, {
    "name": "PCStructuredOcclusionAug",
    "params": {
        "apply_prob": 0.35,
        "range_image_size": [64, 256],
        "blob_count_range": [1, 2],
        "blob_shape_mode": "mixed",
        "rectangle_height_ratio_range": [0.08, 0.18],
        "rectangle_width_ratio_range": [0.08, 0.18],
        "circle_radius_ratio_range": [0.06, 0.12],
        "min_points_kept": 256,
        "keys": ["input_lidar"],
    },
})
dataset, collate_fn = create_dataset(cfg.train_dataset.name, OmegaConf.to_container(cfg.train_dataset.params, resolve=True), pipeline)
batch = collate_fn([dataset[0], dataset[1]])
print(batch["input_lidar"].shape, batch["gt_keypoints"].shape)
PY
```

Loader-level validation with the real HuMMan cross-camera split config:

```bash
uv run python - <<'PY'
from copy import deepcopy
from datasets import *
from misc.registry import create_dataset
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/exp/humman/cross_camera_split/hpe.yml")
pipeline = deepcopy(cfg.train_pipeline)
pipeline.insert(1, {
    "name": "PCStructuredOcclusionAug",
    "params": {
        "apply_prob": 0.35,
        "range_image_size": [64, 256],
        "blob_count_range": [1, 2],
        "blob_shape_mode": "mixed",
        "rectangle_height_ratio_range": [0.08, 0.18],
        "rectangle_width_ratio_range": [0.08, 0.18],
        "circle_radius_ratio_range": [0.06, 0.12],
        "min_points_kept": 256,
        "keys": ["input_lidar"],
    },
})
dataset, collate_fn = create_dataset(cfg.train_dataset.name, OmegaConf.to_container(cfg.train_dataset.params, resolve=True), pipeline)
batch = collate_fn([dataset[0], dataset[1]])
print(batch["input_lidar"].shape, batch["gt_keypoints"].shape)
PY
```
