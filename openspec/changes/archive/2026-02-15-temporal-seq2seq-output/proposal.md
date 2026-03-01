## Why

Current rerun visualization behavior mixes model input assumptions and frame-display behavior, which makes temporal inspection confusing and inconsistent across scripts. We need visualization logic that respects config-defined model input (`seq_len`/pipeline) while letting CLI args control which frames are displayed.

## What Changes

- Keep model prediction behavior unchanged; this change is visualization/inference-script behavior only.
- Standardize visualization scripts to:
  - build model inputs strictly from dataset/config contract (`seq_len`, pipeline, modality/view selection),
  - apply CLI frame controls only to visualization sampling/timeline.
- Ensure multi-frame visualization consistently logs:
  - per-frame input RGB/depth/pointcloud,
  - per-frame GT temporal labels when available (`gt_keypoints_seq`, `gt_smpl_params_seq`),
  - predicted outputs without silently reinterpreting model output semantics.
- Align `visualize_inference_rerun.py` temporal loading behavior with `visualize_sam3d_body_rerun.py` so the same sample/window/frame choices produce comparable timelines.
- Add clear runtime metadata in rerun logs (`sample_id`, source frame indices, frame count) for reproducibility.

## Capabilities

### New Capabilities
- `config-aware-visualization-sampling`: Visualization-time frame selection that remains consistent with config-defined sequence input requirements.

### Modified Capabilities
- `rerun-visualization-pipeline`: Update requirements for config-driven input-window loading, explicit temporal frame sampling, and consistent multi-frame rerun timelines.

## Impact

- Affected code: visualization scripts and rerun helper modules (`scripts/visualize_inference_rerun.py`, `scripts/visualize_sam3d_body_rerun.py`, `scripts/rerun_utils/*`) plus dataset temporal-GT fields used by visualization.
- Affected configs: demo configs used for rerun visualization (HuMMan SAM3D and MMHPE inference visualization).
- Runtime/log impact: rerun `.rrd` files become temporally interpretable and reproducible without changing model training/checkpoint compatibility.
