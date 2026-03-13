## Purpose

`tools/generate_panoptic_sam3_segmentation_masks.py` generates reusable binary person masks for the preprocessed Panoptic dataset using the same SAM3 prompt path used by the repository's SAM-3D-Body integration.

The script is a direct filesystem tool. It does not use dataset configs and does not write experiment artifacts under `logs/`. Instead, it writes sequence-local masks next to the dataset content.

## Input Layout

The canonical input layout is:

```text
<data_root>/<sequence>/rgb/<camera_name>/<image files>
```

Example:

```text
/opt/data/panoptic_kinoptic_single_actor_cropped/161029_flute1/rgb/kinect_1/00000180.jpg
```

## Output Layout

For each processed sequence and camera, masks are written under:

```text
<data_root>/<sequence>/sam_segmentation_mask/<camera_name>/
```

There is one output mask per RGB image.

Filename mapping:
- RGB `.png` -> keep the full filename
- RGB `.jpg` / `.jpeg` / other lossy formats -> use the same basename stem with `.png`

Examples:

```text
rgb/kinect_1/00000180.png -> sam_segmentation_mask/kinect_1/00000180.png
rgb/kinect_1/00000180.jpg -> sam_segmentation_mask/kinect_1/00000180.png
```

Masks are stored as binary lossless images with pixel values `0` and `255`.

## Segmentation Behavior

The script reuses `third_party/sam-3d-body/tools/build_sam.py` and constructs:

```python
HumanSegmentor(name="sam3", ...)
```

That path already prompts SAM3 with:

```text
person
```

Behavior per image:
- run SAM3 text-prompt segmentation
- collect all confident person masks returned by SAM3
- union them into one binary mask
- write one output file for the RGB image

If SAM3 returns no confident person masks, the script writes an all-zero binary mask for that image so the one-image-to-one-mask contract is preserved.

## CLI

Basic usage:

```bash
uv run python tools/generate_panoptic_sam3_segmentation_masks.py \
  --data-root /opt/data/panoptic_kinoptic_single_actor_cropped \
  --sequences 161029_flute1 \
  --cameras kinect_1 \
  --segmentor-path /opt/data/SAM3_checkpoint
```

Process multiple sequences and cameras:

```bash
uv run python tools/generate_panoptic_sam3_segmentation_masks.py \
  --data-root /opt/data/panoptic_kinoptic_single_actor_cropped \
  --sequences 161029_flute1,170915_office1 \
  --cameras kinect_1,kinect_4 \
  --segmentor-path /opt/data/SAM3_checkpoint
```

Overwrite existing masks:

```bash
uv run python tools/generate_panoptic_sam3_segmentation_masks.py \
  --data-root /opt/data/panoptic_kinoptic_single_actor_cropped \
  --sequences 161029_flute1 \
  --cameras kinect_1 \
  --segmentor-path /opt/data/SAM3_checkpoint \
  --overwrite
```

Continue after image-level failures and save a JSON summary:

```bash
uv run python tools/generate_panoptic_sam3_segmentation_masks.py \
  --data-root /opt/data/panoptic_kinoptic_single_actor_cropped \
  --sequences 161029_flute1 \
  --cameras kinect_1 \
  --segmentor-path /opt/data/SAM3_checkpoint \
  --continue-on-error \
  --summary-json logs/panoptic_sam3_masks/flute1_k1_summary.json
```

Run one shard out of four:

```bash
uv run python tools/generate_panoptic_sam3_segmentation_masks.py \
  --data-root /opt/data/panoptic_kinoptic_single_actor_cropped \
  --segmentor-path /opt/data/SAM3_checkpoint \
  --num-shards 4 \
  --shard-index 0 \
  --continue-on-error \
  --summary-json logs/panoptic_sam3_masks/shard_0_of_4.json
```

Launch four disjoint workers in parallel:

```bash
for shard in 0 1 2 3; do
  uv run python tools/generate_panoptic_sam3_segmentation_masks.py \
    --data-root /opt/data/panoptic_kinoptic_single_actor_cropped \
    --segmentor-path /opt/data/SAM3_checkpoint \
    --num-shards 4 \
    --shard-index "${shard}" \
    --continue-on-error \
    --summary-json "logs/panoptic_sam3_masks/shard_${shard}_of_4.json" \
    > "logs/panoptic_sam3_masks/shard_${shard}_of_4.log" 2>&1 &
done
wait
```

## Runtime Behavior

Default behavior:
- strict startup validation for dataset root, CUDA availability when requested, and SAM3 checkpoint presence
- skip existing masks
- stop on the first sequence/camera/image failure

Optional behavior:
- `--overwrite`: regenerate existing output masks
- `--continue-on-error`: record failures and continue processing later items
- `--summary-json`: write the final run summary to JSON
- `--num-shards` / `--shard-index`: process one deterministic subset of sorted `(sequence, camera)` pairs

Progress is shown with `tqdm` at the sequence-camera image loop level.

## Sharding Model

Parallel execution is based on sorted `(sequence, camera)` pairs.

The script:
- enumerates all selected sequence-camera pairs
- sorts them deterministically by sequence then camera
- assigns pair `i` to shard `i % num_shards`

This means:
- shards are disjoint
- multiple workers can write safely to the same dataset root
- reruns remain stable as long as the selected sequence/camera set is unchanged

## Dependencies

The script assumes the same SAM3 environment already used elsewhere in this repository:
- `third_party/sam-3d-body`
- `third_party/sam3`
- a local SAM3 checkpoint, usually:
  - `/opt/data/SAM3_checkpoint/sam3.pt`

If `--segmentor-path` points to a directory, the script looks for `sam3.pt` inside that directory.

## Failure Modes

The script fails fast on:
- missing dataset root
- missing sequence directory
- missing `rgb/` directory
- missing requested camera directory
- missing SAM3 checkpoint
- unreadable RGB images
- failed mask writes

With `--continue-on-error`, per-item failures are collected in the final summary and printed at the end of the run.
