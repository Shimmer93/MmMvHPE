# Data Pipeline

This document explains how dataset loading and transform pipelines work in MMHPE.

## Components

- `datasets/data_api.py`:
  `LitDataModule` builds train/val/test/predict datasets from config.
- `datasets/base_dataset.py`:
  shared transform-pipeline loading and default `collate_fn`.
- `datasets/transforms/*`:
  transform classes referenced by YAML `train_pipeline`, `val_pipeline`, `test_pipeline`.

## Available datasets (registry names)

Exported in `datasets/__init__.py`:
- `MMFiDataset`
- `MMFiPreprocessedDataset`
- `H36MDataset`
- `HummanDataset`
- `HummanPreprocessedDataset`
- `HummanPreprocessedDatasetV2`
- `HummanPreprocessedDatasetV3`
- `HummanCameraDatasetV1`
- `PanopticPreprocessedDatasetV1`

## Batch/sample contract

A sample usually contains:
- `sample_id`: unique identifier
- `modalities`: active modalities for the sample
- `input_rgb`, `input_depth`, `input_lidar`, `input_mmwave` (as available)
- `gt_keypoints`: 3D keypoints
- `gt_smpl_params`: SMPL params
- `*_camera`: modality camera metadata
: one camera is a dict, multi-camera is a list of dicts

`BaseDataset.collate_fn` behavior to know:
- if all sample values of a key are tensors, they are stacked.
- if any value is missing (`None`), a Python list is kept instead of tensor stack.
- for missing `*_affine` keys, it inserts identity `4x4` tensors.

This list-vs-tensor behavior is a common source of shape/type bugs in custom heads.

## Temporal + Multi-Camera shape contract

For sequence runs (`seq_len > 1`) and `configs/exp` multi-camera settings:
- one-camera tensors remain unchanged:
: `input_rgb` / `input_depth`: `(T, C, H, W)` after `ToTensor`
: `input_lidar` / `input_mmwave`: `(T, N, C)` after `PCPad` + `ToTensor`
: `gt_camera_<modality>`: `(T, 9)`
- multi-camera tensors use an added leading view axis `V`:
: `input_rgb` / `input_depth`: `(V, T, C, H, W)`
: `input_lidar` / `input_mmwave`: `(V, T, N, C)`
: `gt_camera_<modality>`: `(V, T, 9)`

`SyncKeypointsWithCameraEncoding` supports both single-frame and sequence 3D keypoints:
- if `gt_keypoints` is `(J, 3)`, it is broadcast across all `T` camera steps
- if `gt_keypoints` is `(T, J, 3)`, it is used step-wise

The synced 2D output is sequence-aligned and view-aligned:
- one-camera: `gt_keypoints_2d_rgb` / `gt_keypoints_2d_depth`: `(T, J, 2)`
- multi-camera: `gt_keypoints_2d_rgb` / `gt_keypoints_2d_depth`: `(V, T, J, 2)`

## HummanPreprocessedDatasetV2 notes

`datasets/humman_dataset_v2.py` is the main newer HuMMan preprocessed loader.

Expected folder structure under `data_root`:
- `rgb/`
- `depth/`
- `lidar/` (optional if generating from depth)
- `cameras/<seq>_cameras.json`
- `smpl/<seq>_smpl_params.npz`
- `skl/<seq>_keypoints_3d.npz`

Important behavior:
- can use YAML split definitions with `split_config` and `split_to_use`.
- if no split config, defaults to person-based 80/20 split.
- can convert depth to point cloud on the fly (`convert_depth_to_lidar=True`).
- can skip heavy image loading while keeping geometry labels (`skeleton_only=True`).
- supports sampling multiple cameras per modality with:
: `rgb_cameras_per_sample`, `depth_cameras_per_sample`, `lidar_cameras_per_sample`
- one-camera configs are still valid without any changes.

## HummanPreprocessedDatasetV3 JSON skeleton notes

`datasets/humman_dataset_v3.py` extends V2 with optional JSON skeleton loading:
- `rgb_skeleton_json`:
: loads 2D keypoints into `gt_keypoints_2d_rgb` (normalized to `[-1, 1]`).
- `lidar_skeleton_json`:
: loads 3D keypoints into `gt_keypoints_lidar` by default (key configurable via `lidar_skeleton_key`).

LiDAR JSON coordinate handling is controlled by:
- `lidar_skeleton_coord`: `"new_world"` or `"world"` (default: `"new_world"`).

The loader converts LiDAR JSON keypoints so they are consistent with dataset mode:
- if dataset uses `apply_to_new_world=True`, LiDAR keypoints are output in new-world coordinates.
- if dataset uses `apply_to_new_world=False`, LiDAR keypoints are output in world coordinates.

## Split config usage

`configs/datasets/humman_split_config.yml` supports:
- `random_split`
- `cross_camera_split`
- `cross_subject_split`
- `cross_action_split`

When `test_mode=True`, the dataset reads `val_dataset` section of the selected split entry.

`configs/datasets/panoptic_split_config.yml` supports:
- `random_split` with deterministic sequence-level ratio split (`ratio` + `random_seed`)
- explicit `train_dataset.sequences` / `val_dataset.sequences` inside `random_split`
- `cross_camera_split` (`train_dataset.cameras` / `val_dataset.cameras`)
- `cross_subject_split` (`train_dataset.subjects` / `val_dataset.subjects`)
- `cross_action_split`-style filtering is also supported via `actions` fields

For Panoptic, sequence-level split is applied first, then optional user sequence allowlist filtering (`sequence_allowlist` or `sequence_list_file`).
Panoptic `subjects`/`actions` are parsed from sequence name `<subject>_<action>` (split at first underscore).

## Panoptic preprocessed dataset notes

`datasets/panoptic_preprocessed_dataset_v1.py` loads sequence-preserving outputs from:
- default root: `/opt/data/panoptic_kinoptic_single_actor_cropped`
- per-sequence required files:
: `meta/sync_map.json`
: `meta/cameras_kinect_cropped.json`
: `meta/manifest.json`
: `gt3d/*.npy`

Key behavior:
- strict validation is enabled by default (`strict_validation=true`) and fails on missing artifacts.
- sample keys follow HuMMan v3 training-facing contract (`sample_id`, `modalities`, `input_*`, `*_camera`, `gt_keypoints`, `gt_smpl_params`, `selected_cameras`, etc.).
- camera names are normalized to `kinect_###` style.

Dataset entry templates are provided in:
- `configs/datasets/panoptic_preprocessed_dataset_v1.yml`

## Transform pipeline config

Pipelines are list-of-dicts:

```yaml
train_pipeline:
  - name: DepthToLiDARPC
    params:
      keys: [input_depth]
  - name: VideoResize
    params:
      size: [180, 320]
      keep_ratio: true
      divided_by: 16
      keys: [input_rgb]
  - name: VideoNormalize
    params:
      norm_mode: imagenet
      keys: [input_rgb]
  - name: PCPad
    params:
      num_points: 1024
      pad_mode: repeat
      keys: [input_lidar]
  - name: ToTensor
    params: null
```

If a transform entry includes `prob`, it is wrapped with `RandomApply`.

## Example: use HummanPreprocessedDatasetV2 in config

```yaml
train_dataset:
  name: HummanPreprocessedDatasetV2
  params:
    data_root: /opt/data/humman
    split: train
    split_config: configs/datasets/humman_split_config.yml
    split_to_use: cross_subject_split
    modality_names: [rgb, depth]
    seq_len: 3
    seq_step: 1
    convert_depth_to_lidar: true
```

## Sanity-check dataset quickly

```bash
uv run python - <<'PY'
from argparse import Namespace
from misc.utils import load_cfg
from datasets.data_api import LitDataModule

cfg = load_cfg("configs/dev/humman_smpl_token_v4.yml")
dm = LitDataModule(cfg)
dm.setup("fit")
print("train size:", len(dm.train_dataset))
print("val size:", len(dm.val_dataset))
sample = dm.train_dataset[0]
print("sample keys:", sorted(sample.keys()))
PY
```

Panoptic smoke test on a ready sequence (`161029_piano2`):

```bash
uv run python - <<'PY'
from datasets import PanopticPreprocessedDatasetV1

ds = PanopticPreprocessedDatasetV1(
    data_root="/opt/data/panoptic_kinoptic_single_actor_cropped",
    modality_names=("rgb", "depth"),
    seq_len=1,
    pad_seq=True,
    split_config="configs/datasets/panoptic_split_config.yml",
    split_to_use="random_split",
    sequence_allowlist=["161029_piano2"],
    strict_validation=True,
)
print("dataset size:", len(ds))
sample = ds[0]
print("sample_id:", sample["sample_id"])
print("keys:", sorted(sample.keys()))
print("rgb camera shape:", sample["rgb_camera"]["intrinsic"].shape)
print("depth camera shape:", sample["depth_camera"]["intrinsic"].shape)
print("gt keypoints shape:", sample["gt_keypoints"].shape)
PY
```

Deterministic ratio split check (same split membership across repeated runs):

```bash
uv run python - <<'PY'
from datasets import PanopticPreprocessedDatasetV1

params = dict(
    data_root="/opt/data/panoptic_kinoptic_single_actor_cropped",
    split_config="configs/datasets/panoptic_split_config.yml",
    split_to_use="random_split",
    test_mode=False,
    strict_validation=True,
)
ds1 = PanopticPreprocessedDatasetV1(**params)
ds2 = PanopticPreprocessedDatasetV1(**params)
seqs1 = sorted(ds1.sequence_data.keys())
seqs2 = sorted(ds2.sequence_data.keys())
print("same_membership:", seqs1 == seqs2)
print("train_sequences:", seqs1)
PY
```
