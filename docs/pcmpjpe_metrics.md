# PCMPJPE Metrics

This document describes pelvis-centered MPJPE metrics added to MMHPE.

## Metrics

- `PCMPJPE` (`metrics/mpjpe.py`): For direct keypoint outputs.
- `SMPL_PCMPJPE` (`metrics/smpl_metrics.py`): For SMPL-based outputs.

Both metrics do the following before computing MPJPE:
- align pelvis translation (prediction pelvis to GT pelvis)
- keep rotation mismatch (root/global orientation differences still contribute to error)

## Orientation Convention

### Non-SMPL (`PCMPJPE`)

`PCMPJPE` is translation-only centered and does not apply orientation alignment.
`skeleton_name` and optional torso-joint index parameters are accepted for config compatibility,
but they do not affect metric computation.

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

`SMPL_PCMPJPE` is also translation-only centered and does not require root orientation fields.

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
