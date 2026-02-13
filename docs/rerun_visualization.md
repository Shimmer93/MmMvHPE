# Rerun Inference Visualization

This document covers the rerun visualization scripts for:

- MMHPE model inference (`scripts/visualize_inference_rerun.py`)
- SAM-3D-Body inference (`scripts/visualize_sam3d_body_rerun.py`)

Both scripts now use shared helpers under `scripts/rerun_utils/` and standardized rerun entity namespaces.

## Output Namespaces

- Inputs: `world/inputs/<modality>/view_<i>/...`
- 3D front view: `world/front/...`
- 3D side view: `world/side/...`
- Metadata: `world/info/...`

## Shared Helper Modules

- `scripts/rerun_utils/layout.py`: config-driven modality/view layout, blueprint, and per-view splitting
- `scripts/rerun_utils/session.py`: rerun session init, timeline steps, and world axis logging
- `scripts/rerun_utils/logging3d.py`: point cloud/skeleton/mesh logging helpers
- `scripts/rerun_utils/geometry.py`: coordinate transforms used by viewers
- `scripts/rerun_utils/image.py`: image conversion for display
- `scripts/rerun_utils/smpl.py`: SMPL conversion and skeleton utilities

## MMHPE Visualization Script

### Purpose

Run a trained MMHPE checkpoint, log configured inputs and predictions, and save/serve a rerun recording.

### Example

```bash
uv run --no-sync python scripts/visualize_inference_rerun.py \
  -c configs/dev/humman_fusionformer_gt_pose_mini.yml \
  --checkpoint logs/fusionformer/humman_fusionformer_gt_pose_mini_20260202_220918/last.ckpt \
  --split test \
  --num_samples 0 \
  --no_serve \
  --save_rrd logs/rerun_smoke/mmmvhpe_smoke.rrd
```

Notes:
- `--num_samples 0` is useful to validate config/checkpoint/rerun wiring without iterating dataset samples.
- For full sample rendering, ensure the selected split has valid sample tensors for the pipeline.

## SAM-3D-Body Visualization Script

### Purpose

Load one sample from a config-selected dataset split, run SAM-3D-Body, and log RGB + overlay + 3D predictions to rerun.

### Image format contract

- Input to `rr.Image` is always `(H, W, 3)` `uint8` RGB.
- If dataset tensors are normalized floats, denormalization is applied from config (`vis_denorm_params`) before logging.
- The script can print image stats before logging with `--debug-image-stats`.

### Keypoint topology source

SAM-3D-Body 2D/3D connectivity uses MHR70 metadata from:

- `third_party/sam-3d-body/sam_3d_body/metadata/mhr70.py` (`pose_info.skeleton_info`)

The rerun script maps link names to keypoint indices from the same metadata and uses that mapping for both 2D overlay edges and 3D skeleton edges.

### Required checkpoint files

Under checkpoint root (default `/opt/data/SAM_3dbody_checkpoints/`):

- `model_config.yaml`
- `model.ckpt`
- `mhr_model.pt`

If any file is missing, the script fails fast with an explicit path message.

### Render mode switch

`--render-mode` is explicit:

- `overlay`: skip mesh logging and keep only 2D overlay logging
- `mesh`: require mesh outputs; fail if unavailable
- `auto`: mesh when available, otherwise overlay

GT behavior:
- GT keypoints are logged when `gt_keypoints` exists.
- GT mesh is logged when GT SMPL fields are sufficient and SMPL model can be loaded.
- Availability flags are logged under `world/info/gt_keypoints_available` and `world/info/gt_mesh_available`.

### Multi-frame control

- `--num-frames` controls how many frames from one sample window are visualized (default: `1`).
- `--frame-index >= 0` is treated as a start anchor; the script uses contiguous frames `[frame_index, frame_index + num_frames - 1]` clipped to valid bounds.
- `--frame-index = -1` uses a center-anchored contiguous window.
- Each visualized source frame is logged as one rerun timeline step, and the source index is written to `world/info/source_frame_index`.

### Example commands

```bash
uv run --no-sync python scripts/visualize_sam3d_body_rerun.py \
  -c configs/demo/humman_smpl_v5_kcam_gcn_mix_synth_v3.yml \
  --split train \
  --sample-idx 0 \
  --num-frames 1 \
  --checkpoint-root logs/rerun_smoke/sam3d_ckpt_stub \
  --render-mode overlay \
  --debug-image-stats \
  --no_serve \
  --save_rrd logs/rerun_smoke/sam3d_overlay_nf1.rrd
```

```bash
uv run --no-sync python scripts/visualize_sam3d_body_rerun.py \
  -c configs/demo/humman_smpl_v5_kcam_gcn_mix_synth_v3.yml \
  --split train \
  --sample-idx 0 \
  --num-frames 3 \
  --checkpoint-root logs/rerun_smoke/sam3d_ckpt_stub \
  --render-mode auto \
  --no_serve \
  --save_rrd logs/rerun_smoke/sam3d_auto_nf3.rrd
```

```bash
uv run --no-sync python scripts/visualize_sam3d_body_rerun.py \
  -c configs/demo/humman_smpl_v5_kcam_gcn_mix_synth_v3.yml \
  --split train \
  --sample-idx 0 \
  --num-frames 3 \
  --checkpoint-root logs/rerun_smoke/sam3d_ckpt_stub \
  --render-mode mesh \
  --no_serve \
  --save_rrd logs/rerun_smoke/sam3d_mesh_nf3.rrd
```

## Known Limitations and Debug Notes

- Some dataset configs may include invalid/inconsistent tensors for certain splits/samples. If visualization fails while fetching a sample, switch split/sample or reduce scope to wiring checks.
- `build_input_layout_from_config` is strict: each modality in `modality_names` must have either:
  - `<modality>_cameras_per_sample`, or
  - `<modality>_cameras`
  and counts must be consistent when both are present.
- For checkpoint layout differences, this script requires `mhr_model.pt` directly under checkpoint root. If your machine stores it under `assets/`, point `--checkpoint-root` to a compatible path (or create a local symlink layout for validation).
- If `uv run` tries to re-resolve heavy CUDA dependencies unexpectedly, use `--no-sync` for local smoke runs in an already prepared environment.
