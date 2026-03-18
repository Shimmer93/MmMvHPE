# SAM3D HuMMan Conversion Visualization

This note covers the HuMMan conversion-check script:

- `scripts/visualize_sam3d_humman_smpl24_conversion.py`

The script is for inspecting the HuMMan-side official conversion used in `scripts/run_sam3d_eval_suite.py`:

- raw SAM output in MHR70 joint format
- officially converted SAM output in SMPL24 joint format
- HuMMan GT SMPL24 joints

The script passes the per-sample RGB camera intrinsics into SAM-3D-Body through `cam_int`. That avoids relying on the estimator's default focal-length heuristic when the dataset already provides calibrated intrinsics.

The converted SMPL24 output comes from the official MHR conversion workflow under `third_party/MHR/tools/mhr_smpl_conversion`, not from a local heuristic joint remap.

## Outputs

For one sample it writes:

- `rgb_gt_smpl24_overlay.jpg`
- `rgb_sam_mhr70_overlay.jpg`
- `rgb_sam_smpl24_overlay.jpg`
- `comparison_figure.png`
- `joint_sets.npz`
- `metadata.json`

`comparison_figure.png` contains:

- GT SMPL24 projected on RGB
- raw SAM MHR70 projected on RGB
- converted SAM SMPL24 projected on RGB
- 3D camera-space skeleton comparison

`joint_sets.npz` stores the exact joint arrays used for plotting.

## Example

```bash
uv run python scripts/visualize_sam3d_humman_smpl24_conversion.py \
  --cfg configs/exp/humman/cross_camera_split/hpe.yml \
  --split test \
  --camera kinect_000 \
  --sample-idx 0 \
  --checkpoint-root /opt/data/SAM_3dbody_checkpoints \
  --mhr-root third_party/MHR \
  --smpl-model-path /opt/data/SMPL_NEUTRAL.pkl
```

With SAM3 masking:

```bash
uv run python scripts/visualize_sam3d_humman_smpl24_conversion.py \
  --cfg configs/exp/humman/cross_camera_split/hpe.yml \
  --split test \
  --camera kinect_000 \
  --sample-idx 0 \
  --checkpoint-root /opt/data/SAM_3dbody_checkpoints \
  --mhr-root third_party/MHR \
  --smpl-model-path /opt/data/SMPL_NEUTRAL.pkl \
  --segmentor-name sam3 \
  --segmentor-path /opt/data/SAM3_checkpoint \
  --use-mask
```
