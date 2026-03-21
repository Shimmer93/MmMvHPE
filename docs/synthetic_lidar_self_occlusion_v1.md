# Synthetic LiDAR Self-Occlusion v1

This document describes the `v1` synthetic LiDAR regeneration workflow that adds self-occlusion-aware point clouds on top of the existing synthetic sample roots.

The `v1` workflow:
- reuses saved synthetic mesh and virtual LiDAR pose artifacts
- keeps the original `v0a` LiDAR artifact untouched
- writes a sibling `v1` LiDAR artifact into the same sample
- lets training/export code choose the LiDAR version from config or CLI parameters

## Artifact Layout

For a regenerated sample such as:

```text
/opt/data/coco/synthetic_data/v0a_val2017/ann_000000183125_img_000000185250/
```

the `v1` workflow adds:

```text
arrays/synthetic_lidar_points_sensor_v1.npy
lidar_qc/lidar_v0a_vs_v1.png          # only when QC rendering is enabled
```

and updates `manifest.json` with:
- `artifacts.synthetic_lidar_points_sensor_v1`
- `lidar_artifacts.v0a`
- `lidar_artifacts.v1`
- `available_lidar_versions`
- `default_lidar_version`

The original `artifacts.synthetic_lidar_points_sensor` path remains unchanged and continues to represent `v0a`.

## Depth-Buffer Resolution

In this workflow, `depth-buffer resolution` means the width and height of the rendered depth map used for the visibility pass in the virtual sensor frame.

Example:
- `512x512` means a square rendered depth map with `512 * 512` depth pixels
- `720x720` means a square rendered depth map with `720 * 720` depth pixels
- `1024x1024` means a square rendered depth map with `1024 * 1024` depth pixels

Higher resolution tends to preserve more visible-surface detail, but it also increases the amount of projected depth-buffer work.

## Commands

Regenerate one sample:

```bash
uv run --no-sync python scripts/regenerate_synthetic_lidar_v1.py \
  --synthetic-root /opt/data/coco/synthetic_data/v0a_val2017 \
  --sample-dir ann_000000183125_img_000000185250 \
  --depth-buffer-resolution 720 \
  --save-qc
```

Regenerate a dataset root with resumable processing:

```bash
uv run --no-sync python scripts/regenerate_synthetic_lidar_v1_dataset.py \
  --synthetic-root /opt/data/coco/synthetic_data/v0a_train2017 \
  --resume
```

Benchmark candidate resolutions on a validation subset:

```bash
uv run --no-sync python scripts/benchmark_synthetic_lidar_v1.py \
  --synthetic-root /opt/data/coco/synthetic_data/v0a_val2017 \
  --output-dir logs/synthetic_lidar_v1_benchmark \
  --start-index 0 \
  --max-samples 4 \
  --save-qc
```

Export GT bundles against the regenerated `v1` LiDAR artifact:

```bash
uv run --no-sync python scripts/export_synthetic_gt_formats.py \
  --synthetic-root /opt/data/coco/synthetic_data/v0a_val2017 \
  --sample-dir /opt/data/coco/synthetic_data/v0a_val2017/ann_000000183125_img_000000185250 \
  --lidar-version v1
```

## Training Config Selection

`SyntheticExportedTrainingDataset` now accepts `lidar_version`.

Example:

```yaml
train_dataset:
  name: SyntheticExportedTrainingDataset
  params:
    data_root: /opt/data/coco/synthetic_data/v0a_train2017
    target_format: humman
    lidar_version: v1
```

The synthetic pretrain configs currently keep `lidar_version: 'v0a'` so they remain valid before full `v1` regeneration. Switch those configs to `v1` after the target synthetic root has been regenerated.

## Benchmark Result

Validation benchmark command:

```bash
uv run --no-sync python scripts/benchmark_synthetic_lidar_v1.py \
  --synthetic-root /opt/data/coco/synthetic_data/v0a_val2017 \
  --output-dir logs/synthetic_lidar_v1_benchmark \
  --start-index 0 \
  --max-samples 4 \
  --save-qc
```

Summary file:
- `logs/synthetic_lidar_v1_benchmark/benchmark_summary.json`

QC figures:
- `logs/synthetic_lidar_v1_benchmark/ann_000000183125_img_000000185250_benchmark.png`
- `logs/synthetic_lidar_v1_benchmark/ann_000000183126_img_000000425226_benchmark.png`
- `logs/synthetic_lidar_v1_benchmark/ann_000000183301_img_000000549390_benchmark.png`
- `logs/synthetic_lidar_v1_benchmark/ann_000000183349_img_000000007281_benchmark.png`

Measured mean runtime per sample on the 4-sample validation subset:
- `512x512`: `0.0359s`
- `720x720`: `0.0342s`
- `1024x1024`: `0.0427s`

Estimated runtime for the current synthetic roots using those mean runtimes:
- `val2017` root (`1097` samples):
  - `512x512`: `0.0109h`
  - `720x720`: `0.0104h`
  - `1024x1024`: `0.0130h`
- `train2017` root (`26191` samples):
  - `512x512`: `0.2612h`
  - `720x720`: `0.2486h`
  - `1024x1024`: `0.3103h`

Mean Chamfer distance to the `1024x1024` reference point cloud:
- `512x512`: `0.01117`
- `720x720`: `0.01112`
- `1024x1024`: `0.0`

## Chosen Default

The selected default depth-buffer resolution is `720x720`.

Reason:
- its runtime was slightly better than `512x512` on the benchmark subset
- its Chamfer distance to the `1024x1024` reference was marginally lower than `512x512`
- it keeps a larger visible-point set and denser occupied depth map than `512x512`
- `1024x1024` increased runtime without enough quality gain to justify making it the default for bulk regeneration

## Validation Notes

Validated during implementation:
- one-sample `v1` regeneration with inline QC output
- 8-sample val-root regeneration pass
- synthetic exported dataset loading for both `v0a` and `v1` on HuMMan-style and Panoptic-style validation batches
- one-sample GT export run with `--lidar-version v1`
