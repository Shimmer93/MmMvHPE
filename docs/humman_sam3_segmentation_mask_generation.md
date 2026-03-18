# HuMMan SAM3 Segmentation Mask Generation

`tools/generate_humman_sam3_segmentation_masks.py` generates one SAM3 person mask per RGB image for the flat `humman_cropped` layout.

## Dataset contract

Expected input layout:

- `<data_root>/rgb/<sequence>_kinect_<camera>_<frame>.jpg`

Example:

- `/opt/data/humman_cropped/rgb/p000441_a000701_kinect_008_000045.jpg`

Default output layout:

- `<data_root>/sam_segmentation_mask/<sequence>_kinect_<camera>_<frame>.png`

External output layout with `--mask-root`:

- `<mask_root>/<sequence>_kinect_<camera>_<frame>.png`

The script groups work by `(sequence, camera)` pairs parsed from the RGB filenames so it can shard deterministically, similar to the Panoptic mask generator.

## Behavior

- prompt is always `person`
- all detected person masks are unioned into one binary mask
- existing mask files are skipped unless `--overwrite` is used
- failures stop the run by default; use `--continue-on-error` to continue

## Example

Generate masks for the full dataset:

```bash
uv run python tools/generate_humman_sam3_segmentation_masks.py \
  --data-root /opt/data/humman_cropped \
  --mask-root /opt/data/humman_cropped_masks \
  --segmentor-path /opt/data/SAM3_checkpoint \
  --summary-json logs/humman_sam3_masks/full_summary.json
```

Generate only one sequence-camera subset:

```bash
uv run python tools/generate_humman_sam3_segmentation_masks.py \
  --data-root /opt/data/humman_cropped \
  --mask-root /opt/data/humman_cropped_masks \
  --sequences p000441_a000701 \
  --cameras kinect_008 \
  --segmentor-path /opt/data/SAM3_checkpoint \
  --summary-json logs/humman_sam3_masks/p000441_a000701_k008.json
```

Use two shards:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python tools/generate_humman_sam3_segmentation_masks.py \
  --data-root /opt/data/humman_cropped \
  --mask-root /opt/data/humman_cropped_masks \
  --segmentor-path /opt/data/SAM3_checkpoint \
  --num-shards 2 \
  --shard-index 0
```

Use the shard launcher with an external mask root:

```bash
DATA_ROOT=/opt/data/humman_cropped \
MASK_ROOT=/opt/data/humman_cropped_masks \
SEGMENTOR_PATH=/opt/data/SAM3_checkpoint \
NUM_SHARDS=2 \
DEVICE_IDS=0,1 \
LOG_DIR=logs/humman_sam3_masks \
./scripts/run_humman_sam3_mask_shards.sh
```
