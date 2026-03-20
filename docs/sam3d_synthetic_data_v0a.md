# SAM-3D-Body Synthetic Data v0-a

This document describes the first synthetic sample generation pipeline for MMHPE.

## Goal

`v0-a` is a producer-side validation milestone:
- source dataset: COCO val under `/opt/data/coco`
- input: one RGB image with one selected person instance
- reconstruction: SAM-3D-Body on the full image with a saved full-image mask as auxiliary input
- synthetic output: one pelvis-centered 3D keypoint set, one virtual LiDAR pose, one synthetic LiDAR-style point cloud, and visual QC artifacts

This change does not integrate synthetic data into `main.py` training yet.

## Pipeline Stages

1. Select one eligible COCO person annotation.
2. Rasterize and save the annotation mask in source-image coordinates.
3. Run SAM-3D-Body on the full image with explicit bbox + saved mask.
4. Reject invalid outputs early if mask/reconstruction quality checks fail.
5. Canonicalize 3D outputs with pelvis center defined as midpoint of MHR70 left/right hips.
6. Sample one virtual LiDAR pose around the canonical body.
7. Sample visible-facing mesh surface points and transform them into sensor frame.
8. Save arrays, metadata manifest, and optional visualization artifacts.

## Output Layout

Per sample, outputs are written under:

`logs/synthetic_data/v0a/ann_<annotation_id>_img_<image_id>/`

Typical files:
- `source_mask.png`
- `manifest.json`
- `arrays/pred_keypoints_3d_raw.npy`
- `arrays/pred_vertices_raw.npy`
- `arrays/pred_keypoints_3d_canonical.npy`
- `arrays/pred_vertices_canonical.npy`
- `arrays/pelvis_source_frame.npy`
- `arrays/synthetic_lidar_points_sensor.npy`
- `arrays/mesh_faces.npy`
- `arrays/lidar_extrinsic_world_to_sensor.npy`

Optional files when enabled:
- `source_rgb.png`
- `sam3d_overlay.png`
- `summary.png`

## Run One Sample

```bash
uv run --no-sync python scripts/run_sam3d_synthetic_v0a.py \
  --data-root /opt/data/coco \
  --checkpoint-root /opt/data/SAM_3dbody_checkpoints \
  --output-dir logs/synthetic_data/v0a \
  --start-index 0 \
  --max-samples 1
```

Enable replicated source RGB and visualization artifacts explicitly:

```bash
uv run --no-sync python scripts/run_sam3d_synthetic_v0a.py \
  --data-root /opt/data/coco \
  --checkpoint-root /opt/data/SAM_3dbody_checkpoints \
  --output-dir logs/synthetic_data/v0a_vis \
  --start-index 0 \
  --max-samples 1 \
  --save-source-rgb \
  --save-visualizations
```

## Run a Small Batch

```bash
uv run --no-sync python scripts/run_sam3d_synthetic_v0a.py \
  --data-root /opt/data/coco \
  --checkpoint-root /opt/data/SAM_3dbody_checkpoints \
  --output-dir logs/synthetic_data/v0a_batch \
  --start-index 0 \
  --max-samples 8
```

## Run a Full COCO Split

Process the local val split:

```bash
uv run --no-sync python scripts/run_sam3d_synthetic_v0a_coco_dataset.py \
  --data-root /opt/data/coco \
  --split val2017 \
  --checkpoint-root /opt/data/SAM_3dbody_checkpoints \
  --output-dir logs/synthetic_data/v0a_val2017 \
  --resume
```

Process the training split after it finishes downloading:

```bash
uv run --no-sync python scripts/run_sam3d_synthetic_v0a_coco_dataset.py \
  --data-root /opt/data/coco \
  --split train2017 \
  --checkpoint-root /opt/data/SAM_3dbody_checkpoints \
  --output-dir logs/synthetic_data/v0a_train2017 \
  --resume
```

Each dataset run writes:
- one per-sample `manifest.json` under the sample directory
- `run_results_<split>_<start>_<end>.jsonl`
- `run_summary_<split>_<start>_<end>.json`

By default, dataset-scale generation avoids saving replicated source RGB images and visualization artifacts to reduce storage. Use `--save-source-rgb` and `--save-visualizations` only when needed for inspection.

## Current Quality Gates

- only non-crowd COCO person annotations are accepted
- minimum person annotation area and visible keypoints are enforced
- mask pixel count must exceed threshold
- SAM-3D-Body output tensors must be finite
- reconstructed mesh must contain enough vertices

## Limitations

- only COCO val is wired in `v0-a`
- no fallback mask generator yet; mask provenance is currently COCO annotation segmentation
- no crop-first path by design
- no large-scale throughput work
- no beam-accurate LiDAR simulation; point clouds are visible-facing surface samples
- no training integration into MMHPE loaders yet

## Follow-Up Milestones

- `v0-b`: better sample filtering and larger small-batch review workflow
- `v0-c`: exporter or dataset adapter into MMHPE-compatible training format
- `v1`: stronger LiDAR realism, richer sensor distributions, and broader source datasets

## Target-Format Export

Synthetic generation now has a follow-on export stage for training-facing GT bundles.

The exporter:
- reads an existing synthetic sample directory
- reruns SAM-3D-Body to recover camera-side outputs not saved by `v0-a`
- writes minimal HuMMan-compatible and Panoptic-compatible GT bundles under `exports/`
- avoids duplicating existing source artifacts and point clouds

Example:

```bash
uv run --no-sync python scripts/export_synthetic_gt_formats.py \
  --synthetic-root /opt/data/coco/synthetic_data/v0a_train2017 \
  --sample-dir /opt/data/coco/synthetic_data/v0a_train2017/ann_000000183014_img_000000045687 \
  --mhr-repo-root /tmp/MHR \
  --smpl-model-path weights/smpl/SMPL_NEUTRAL.pkl
```

See [synthetic_target_format_export.md](/home/yzhanghe/MmMvHPE/docs/synthetic_target_format_export.md) for the full contract and output layout.
