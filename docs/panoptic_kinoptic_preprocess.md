# Panoptic Kinoptic Single-Actor Preprocessing

This document describes `tools/preprocess_panoptic_kinoptic.py`, a sequence-preserving preprocessing script for single-actor Panoptic Kinoptic data.

## Purpose

- reduce training-time I/O and decode overhead,
- produce compact cropped RGB and depth frames,
- synchronize body annotations with Kinect streams using sequence timing metadata,
- keep all outputs grouped by sequence.

## Required Input Per Sequence

Under `<root>/<sequence>/`:

- `ksynctables_<sequence>.json`
- `kcalibration_<sequence>.json`
- `hdPose3d_stage1_coco19/body3DScene_*.json`
- `kinectVideos/kinect_50_XX.mp4` (at least one Kinect node)

## Synchronization Rule

- Body frames come from `body3DScene_*.json` with `univTime`.
- For each Kinect node, nearest color/depth timestamps are selected from `ksynctables`.
- Matches are kept only when both color/depth deltas are within `--max-sync-delta-ms`.
- Frames outside tolerance are dropped.

## Cropping Rule

- RGB crops are computed per `(sequence, camera)` in a HuMMan-cropped style:
  - run YOLO person detection on sampled synchronized frames,
  - take union bbox,
  - convert to square crop,
  - crop+resize to configured RGB output size.
- If no detection is found, full-frame square crop is used.

## Output Layout

Per sequence, outputs are written to:

- `<out_root>/<sequence>/rgb/kinect_<node>/<body_frame_id>.jpg`
- `<out_root>/<sequence>/depth/kinect_<node>/<body_frame_id>.png`
- `<out_root>/<sequence>/gt3d/<body_frame_id>.npy` (19x4 float16, from `joints19`)
- `<out_root>/<sequence>/meta/sync_map.json`
- `<out_root>/<sequence>/meta/crop_params.json`
- `<out_root>/<sequence>/meta/cameras_kinect_cropped.json` (intrinsics adjusted for crop+resize)
- `<out_root>/<sequence>/meta/manifest.json`

No frames are mixed across sequences.

## Split Semantics (Preprocessed Dataset Loading)

The preprocessed Panoptic dataset loader (`PanopticPreprocessedDatasetV1`) supports
multiple split modes from `configs/datasets/panoptic_split_config.yml`.

- `temporal_split`:
  - preferred for sequence-preserving train/val splits;
  - each selected sequence is split by synchronized frame order;
  - train uses the first `floor(ratio * N)` frames and val/test uses the tail.
- `cross_subject_split`:
  - explicit sequence-level partition using configured `sequences` lists;
  - use this when you want whole sequences assigned to train vs val/test.
- `random_split`:
  - legacy sequence-level deterministic random partition;
  - kept for backward compatibility, but `temporal_split` + explicit
    `cross_subject_split` is clearer and recommended for Panoptic.

Migration from legacy Panoptic `random_split`:
- if you want per-sequence temporal head/tail splitting, switch to `split_to_use: temporal_split`;
- if you want whole-sequence partitioning, use `split_to_use: cross_subject_split`
  and set explicit `train_dataset.sequences` / `val_dataset.sequences`.

## Commands

Smoke test (`161029_piano2`, bounded frames):

```bash
uv run python tools/preprocess_panoptic_kinoptic.py \
  --root-dir /data/shared/panoptic-toolbox \
  --out-dir /tmp/panoptic_preprocess_smoke \
  --sequences 161029_piano2 \
  --max-body-frames 64 \
  --continue-on-error
```

List-driven run:

```bash
uv run python tools/preprocess_panoptic_kinoptic.py \
  --root-dir /data/shared/panoptic-toolbox \
  --out-dir /opt/data/panoptic_kinoptic_single_actor_cropped \
  --sequence-list /tmp/panoptic_single_actor_sequences.txt \
  --continue-on-error
```

Example: visualize with per-sequence temporal split:

```bash
uv run python scripts/visualize_inference_rerun.py \
  --cfg configs/vis/panoptic_seq3_piano3.yml \
  --checkpoint logs/occlusion/panoptic_seq3_20260302_191205/PanopticHPE-epoch=10-val_mpjpe=0.0811.ckpt \
  --split train \
  --num_samples 8 \
  --device cuda \
  --no_serve \
  --save_rrd /tmp/panoptic_temporal_split.rrd \
  --skeleton_format panopticcoco19
```

Example: use explicit sequence-level split (cross-subject lists in split config):

```bash
uv run python main.py \
  -c configs/exp/panoptic/cross_camera_split/hpe.yml \
  -g 1 -n 1 -w 4 -b 2 -e 1 -p 2 --pin_memory \
  --exp_name panoptic_cross_subject_explicit \
  --version smoke_$(date +'%Y%m%d_%H%M%S')
```
