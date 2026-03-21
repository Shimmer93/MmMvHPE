# HuMMan Segmentation Mask Runtime Transforms

## Purpose

`ApplyHummanSegmentationMask` applies existing HuMMan SAM3 binary masks to RGB, depth, and/or LiDAR inputs at dataset runtime, without writing a second masked dataset to disk.

The transform is HuMMan-specific and expects the flat mask layout produced by `tools/generate_humman_sam3_segmentation_masks.py`:

```text
<mask_root>/<sequence>_<camera>_<frame>.png
```

Example:

```text
/opt/data/humman_cropped_masks/p000441_a000701_kinect_008_000045.png
```

## Implementation

Files:

- `datasets/transforms/humman_mask_transforms.py`
- `datasets/humman_dataset_v2.py`

The dataset now exposes two extra sample fields used by the transform:

- `unit`: dataset sample unit
- `selected_frame_ids`: exact frame ids per selected camera and modality

`selected_frame_ids` is required because HuMMan mask filenames use the real frame number from disk, not the dataset window index.

## Behavior

The transform:

- accepts `apply_to: ['rgb']`, `['depth']`, `['lidar']`, or combinations such as `['rgb', 'lidar']`
- resolves masks from `seq_name`, selected camera names, and `selected_frame_ids`
- zeroes masked-out RGB pixels directly in the RGB image plane
- reprojects the RGB mask into the depth image plane before zeroing depth pixels
- projects LiDAR points into the selected RGB mask view and drops background points
- raises explicit errors on missing masks, unreadable masks, unresolved metadata, and shape mismatch

For depth and LiDAR reprojection, the transform currently requires:

- `unit: m`
- RGB camera metadata present in the sample

That matches the default HuMMan dataset configuration already used in this repository.

## Usage

Insert the transform before normalization / formatting steps in a HuMMan pipeline.

External mask root example:

```yaml
- name: ApplyHummanSegmentationMask
  params:
    apply_to: ['rgb', 'lidar']
    mask_root: /opt/data/humman_cropped_masks
- name: VideoNormalize
  params:
    norm_mode: imagenet
    keys: [input_rgb]
- name: ToTensor
  params: null
```

If masks are stored under `<data_root>/sam_segmentation_mask/`, you can omit `mask_root` and rely on the dataset root instead.

## Validation

Syntax check:

```bash
uv run python -m py_compile \
  datasets/humman_dataset_v2.py \
  datasets/transforms/humman_mask_transforms.py
```

Minimal runtime smoke test:

```bash
uv run python - <<'PY'
from datasets import HummanPreprocessedDatasetV2

ds = HummanPreprocessedDatasetV2(
    data_root="/opt/data/humman_cropped",
    modality_names=("rgb", "depth"),
    split="train_mini",
    seq_len=1,
    rgb_cameras=["kinect_000"],
    depth_cameras=["kinect_000"],
    convert_depth_to_lidar=True,
    pipeline=[
        {
            "name": "ApplyHummanSegmentationMask",
            "params": {
                "apply_to": ["rgb", "lidar"],
                "mask_root": "/opt/data/humman_cropped_masks",
            },
        },
    ],
)
sample = ds[0]
print("keys:", sorted(k for k in sample.keys() if k.startswith("input_") or k.endswith("_camera")))
PY
```
