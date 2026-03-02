# PCMPJPE Metrics

This document describes pelvis-centered MPJPE metrics added to MMHPE.

## Metrics

- `PCMPJPE` (`metrics/mpjpe.py`): For direct keypoint outputs.
- `SMPL_PCMPJPE` (`metrics/smpl_metrics.py`): For SMPL-based outputs.

Both metrics do the following before computing MPJPE:
- align pelvis translation (prediction pelvis to GT pelvis)
- align pelvis orientation (root rotation) so articulated pose error is measured in a matched root frame

## Orientation Convention

### Non-SMPL (`PCMPJPE`)

`PCMPJPE` derives root orientation from keypoints using the same convention as
`PanopticPreprocessedDatasetV1._estimate_root_rotation_from_joints19`:

- `x` axis: right hip minus left hip
- `y` seed: neck minus body center
- `z` axis: cross(`x`, `y_seed`)
- re-orthogonalized `y` axis: cross(`z`, `x`)

Default indices:
- neck: 0
- body center: 2
- left hip: 6
- right hip: 12

The metric resolves these joints from `misc/skeleton.py` via `skeleton_name` and raises
explicit `ValueError` if required joints are missing or vectors are degenerate.

Supported `skeleton_name` values include:
- `smpl`
- `h36m`
- `mmbody`
- `panoptic_coco19`
- `coco`
- `simple_coco`
- `itop`
- `milipoint`

You can override inferred indices explicitly with:
- `neck_idx`
- `bodycenter_idx`
- `lhip_idx`
- `rhip_idx`

### SMPL (`SMPL_PCMPJPE`)

`SMPL_PCMPJPE` uses explicit root rotation (`global_orient`) from SMPL outputs.

Priority for prediction root rotation:
1. `preds['pred_smpl']['global_orient']`
2. first 3 dims of `preds['pred_smpl_params']`

Priority for GT root rotation:
1. `targets['gt_smpl']['global_orient']`
2. first 3 dims of `targets['gt_smpl_params']`

If required fields are unavailable, the metric fails fast with an explicit error.

## Config Example

```yaml
metrics:
  - name: 'SMPL_PCMPJPE'
    params:
      pelvis_idx: 0
  - name: 'PCMPJPE'
    params:
      use_smpl: false
      affix: no_smpl
      skeleton_name: 'smpl'
      pelvis_idx: 0
    alias: 'PCMPJPE_no_smpl'
```

## Run Example

```bash
uv run python main.py --config configs/dev/humman_leir_rgbpc.yml --task test
```

Evaluation logs include `test_smpl_pcmpjpe` and/or `test_pcmpjpe_no_smpl` when configured.
