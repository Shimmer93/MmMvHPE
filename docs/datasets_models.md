# Datasets and Models

This document covers the dataset and model components currently used for multi-modal human pose estimation:

- `datasets/humman_dataset_v2.py`
- `models/aggregators/trans_aggregator_v4.py`
- `models/heads/regression_head_v5.py`
- `models/heads/vibe_token_head_v5.py`
- `models/heads/vggt_camera_head_v5.py`

## Dataset: HummanPreprocessedDatasetV2

**File:** `datasets/humman_dataset_v2.py`  
**Purpose:** Load preprocessed HuMMan sequences with RGB, depth, and/or lidar modalities and align them to a pelvis-centered world frame.

### Expected data layout

Under `data_root`, this dataset expects the following folders:

- `rgb/` frames (per-camera RGB images)
- `depth/` frames (per-camera depth images)
- `lidar/` frames (per-camera point clouds, `.npy`)
- `cameras/` JSON camera parameters per sequence
- `smpl/` SMPL params per sequence (`*_smpl_params.npz`)
- `skl/` precomputed 3D keypoints (`*_keypoints_3d.npz`)

Camera JSON files must be named `cameras/<seq_name>_cameras.json` and include `K`, `R`, `T` entries per camera.  
SMPL files must be named `smpl/<seq_name>_smpl_params.npz` and include `global_orient`, `body_pose`, `betas`, `transl`.

### Key behaviors

- **Split logic:** default is an 80/20 person-ID split (with `train_mini`/`test_mini`), or use `split_config` + `split_to_use` to load predefined subject/action/camera splits.
- **Sequence sampling:** samples are sliding windows of length `seq_len`, advanced by `seq_step`.
- **Pelvis-centered frame:** `gt_keypoints` and camera extrinsics are transformed into a new world where the pelvis is at the origin and SMPL global orientation is zeroed.
- **Depth-to-lidar conversion:** when `convert_depth_to_lidar=True` and `lidar` is not explicitly requested, depth frames are converted into point clouds and stored as `input_lidar`.
- **Modalities list:** `sample["modalities"]` is updated to reflect actual inputs (e.g., depth swapped for lidar).

### Constructor parameters (high level)

- `data_root`: dataset root directory.
- `split`: `train`, `test`, `train_mini`, `test_mini`, or `all`.
- `split_config`: path to a split YAML (e.g., `configs/datasets/humman_split_config.yml`).
- `split_to_use`: split key inside the YAML (e.g., `random_split`, `cross_subject_split`).
- `test_mode`: when true, uses the YAML `val_dataset` split instead of `train_dataset`.
- `modality_names`: tuple of requested modalities (`rgb`, `depth`, `lidar`).
- `rgb_cameras`, `depth_cameras`: optional camera filters.
- `seq_len`, `seq_step`, `pad_seq`, `causal`: sequence sampling controls.
- `use_all_pairs`: if true, returns all RGB/depth camera pairs per window.
- `colocated`: force RGB and depth cameras to match.
- `convert_depth_to_lidar`: generate lidar point clouds from depth frames.

### Sample output fields

Common keys:

- `sample_id`: unique ID per window and camera pairing.
- `modalities`: list of active modalities after any conversions.
- `gt_keypoints`: `(24, 3)` pelvis-centered keypoints (float32).
- `gt_smpl_params`: `(82,)` pose (72) + betas (10), root rotation zeroed.

Modality-specific:

- `input_rgb`: list of `seq_len` RGB frames `(H, W, 3)`.
- `input_depth`: list of `seq_len` depth frames `(H, W)` (meters when `unit="m"`).
- `input_lidar`: list of `seq_len` point clouds `(N, 3)`.
- `rgb_camera` / `depth_camera` / `lidar_camera`: dict with `intrinsic (3x3)` and `extrinsic (3x4)`.

## Model: TransformerAggregatorV4

**File:** `models/aggregators/trans_aggregator_v4.py`  
**Purpose:** Fuse modality-specific token streams with a configurable attention schedule.

### Inputs

The `forward` method expects a tuple:

```
features = (features_rgb, features_depth, features_lidar, features_mmwave)
```

Each non-`None` tensor should be shaped `(B, T, N, C)` where:

- `B`: batch size
- `T`: temporal length
- `N`: patch/token count per modality
- `C`: feature dim (matches `input_dims` per modality)

### Token layout

For each modality, the aggregator appends special tokens in this order:

1. camera token (1)
2. register tokens (`num_register_tokens`)
3. SMPL tokens (`num_smpl_tokens`)
4. joint tokens (`num_joints`, default 24)

Modalities also receive a learned modality embedding.

### Attention schedule

The `aa_order` controls the sequence of attention blocks applied in each stage:

- `single`: self-attention within each modality
- `cross_modality`: attention over merged patch tokens across modalities
- `cross_joint`: cross-attention between pose tokens and full modality tokens
- `joint_to_camera`: update camera tokens using joint tokens
- `gcn`: temporal GCN over joint tokens using the SMPL skeleton adjacency

`aa_block_size` determines how many blocks of each type are applied per stage, and `depth` must be divisible by it.

### Output

Returns a list of intermediate token stacks (one per block). Each stack is shaped:

```
(B, T, M, S, D)
```

Where:

- `M`: number of active modalities
- `S`: number of special tokens (camera + register + SMPL + joints)
- `D`: `embed_dim`

These outputs are consumed by head modules such as `RegressionKeypointHeadV5` and `VIBETokenHeadV5`.

## Model Head: RegressionKeypointHeadV5

**File:** `models/heads/regression_head_v5.py`  
**Purpose:** Predict 2D/3D keypoints from joint tokens and compute per-modality + global losses.

### Inputs

- `x`: token stacks from the aggregator. Can be:
  - a single tensor `(B, T, M, S, D)`, or
  - a list of tensors (one per layer), in which case `last_n_layers` controls which are used.

### Outputs

```
{
  "per_modality": [pred_m0, pred_m1, ...],
  "global": pred_global
}
```

- Per-modality predictions are `(B, J, 2)` for `rgb`/`depth`, and `(B, J, 3)` for `lidar`/`mmwave`.
- Global prediction is always `(B, J, 3)`.

### Key details

- Uses joint tokens and camera tokens; global prediction concatenates both per modality.
- When `modalities` are provided in the batch, it automatically switches between 2D and 3D regression heads.
- `loss` projects keypoints into the image plane for RGB/depth using `gt_camera_*` pose encodings and compares against normalized 2D targets in `[-1, 1]`.

## Model Head: VIBETokenHeadV5

**File:** `models/heads/vibe_token_head_v5.py`  
**Purpose:** Regress SMPL pose/shape and keypoints from SMPL + joint tokens using a VIBE-style iterative regressor.

### Inputs

- `x`: token stacks from the aggregator; can be a list of layer outputs or a single tensor.

### Outputs

```
{
  "global": {
    "pred_smpl_params": (B, 82),
    "pred_keypoints": (B, 24, 3),
    "pred_rotmat": (B, 24, 3, 3)
  },
  "per_modality": { ... }  # only if return_per_modality=True
}
```

### Key details

- Uses the last temporal step (`T` index -1) and extracts SMPL + joint tokens.
- The regressor iteratively updates pose and shape (`n_iters`) starting from `smpl_mean_params`.
- Global fusion concatenates modalities into a single embedding; optional per-modality regression is supported.
- Loss terms are keyed by substring (`keypoint`, `smplpose`, `smplshape`, `rotmat`) and apply to global outputs.

## Model Head: VGGTCameraHeadV5

**File:** `models/heads/vggt_camera_head_v5.py`  
**Purpose:** Predict per-modality camera pose encodings from camera tokens and compute camera + projection losses.

### Inputs

- `x`: token stacks from the aggregator; can be a list of layer outputs or a single tensor.
- `feature_slice`: controls which half of concatenated features are used (`full`, `first_half`, `second_half`).
- `last_n_layers`: if set, concatenates only the last N layers before prediction.

### Outputs

- `forward` returns a list of pose encodings, one per refinement iteration.
- `predict` returns only the last iteration output.

Each pose encoding has shape `(B, M, 9)` for `absT_quaR_FoV` (translation + quaternion + FoV).

### Key details

- Uses only the camera token per modality (`S=1` in `(B, M, S, C)` token layout).
- `loss` compares predicted encodings to `gt_camera_<modality>` via `CameraLoss`.
- Projection losses for `rgb`/`depth` use GT intrinsics (from `gt_camera_<modality>`) and image sizes from inputs; keypoints are normalized to `[-1, 1]`.
- `lidar`/`mmwave` projection losses compare 3D points in camera coordinates.
