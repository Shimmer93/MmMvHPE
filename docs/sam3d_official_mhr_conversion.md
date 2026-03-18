# Official SAM3D MHR-to-SMPL Conversion

This note covers the official HuMMan SAM3D conversion wrapper:

- `misc/official_mhr_smpl_conversion.py`

The wrapper integrates the upstream MHR conversion implementation from:

- `third_party/MHR`
- `third_party/MHR/tools/mhr_smpl_conversion`

It is used by:

- `scripts/run_sam3d_eval_suite.py` for HuMMan evaluation
- `scripts/visualize_sam3d_humman_smpl24_conversion.py` for HuMMan visual inspection

## What the wrapper does

The wrapper:

- validates the vendored MHR checkout exists
- validates required conversion assets exist
- validates the SMPL model file exists
- validates the runtime has the required `smplx` package and Meta `pymomentum` package
- constructs the upstream `MHR` model and `Conversion` object
- converts SAM3D outputs to fitted SMPL outputs
- returns SMPL24 joints, optional vertices, and fitting errors

The wrapper does not silently fall back to the old heuristic joint remap. If the official dependency stack is incomplete, it raises an explicit error.

## Required assets

The MHR checkout must include:

- `third_party/MHR/assets/compact_v6_1.model`
- `third_party/MHR/assets/lod1.fbx`
- `third_party/MHR/assets/corrective_blendshapes_lod1.npz`
- `third_party/MHR/assets/corrective_activation.npz`

The conversion tool must include:

- `third_party/MHR/tools/mhr_smpl_conversion/assets/head_hand_mask.npz`
- `third_party/MHR/tools/mhr_smpl_conversion/assets/mhr_face_mask.ply`
- `third_party/MHR/tools/mhr_smpl_conversion/assets/subsampled_vertex_indices.npy`
- `third_party/MHR/tools/mhr_smpl_conversion/assets/mhr2smpl_mapping.npz`

The SMPL model path used by MMHPE defaults to:

- `/opt/data/SMPL_NEUTRAL.pkl`

To populate the vendored MHR asset folder:

```bash
cd third_party/MHR
curl -OL https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip
unzip assets.zip
rm assets.zip
```

## Dependency note

The wrapper expects Meta's `pymomentum` package with modules such as:

- `pymomentum.geometry`
- `pymomentum.torch.character`

The generic PyPI `pymomentum` package is not sufficient.

## Example commands

HuMMan evaluation:

```bash
uv run python scripts/run_sam3d_eval_suite.py \
  --scenarios humman_cross_camera_test \
  --checkpoint-root /opt/data/SAM_3dbody_checkpoints \
  --mhr-root third_party/MHR \
  --smpl-model-path /opt/data/SMPL_NEUTRAL.pkl
```

HuMMan visualization:

```bash
uv run python scripts/visualize_sam3d_humman_smpl24_conversion.py \
  --cfg configs/exp/humman/cross_camera_split/hpe.yml \
  --split test \
  --camera kinect_001 \
  --sample-idx 0 \
  --checkpoint-root /opt/data/SAM_3dbody_checkpoints \
  --mhr-root third_party/MHR \
  --smpl-model-path /opt/data/SMPL_NEUTRAL.pkl
```
