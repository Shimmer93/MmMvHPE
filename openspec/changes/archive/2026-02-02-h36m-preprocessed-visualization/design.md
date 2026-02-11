## Context

We already have `scripts/visualize_h36m_preprocessed_data.py`, but it only shows RGB (if present)
and 3D skeleton plots. There is no dedicated multiview overlay of 2D keypoints on each RGB view,
which is the key debugging tool for verifying 3D→2D projection correctness in the preprocessed
H36M dataset. The new script must reuse existing dataset and visualization utilities without adding
dependencies, and must work with config-defined pipelines.

## Goals / Non-Goals

**Goals:**
- Provide a script under `scripts/` that loads a specified training config and split.
- Overlay N-view 2D keypoints on their corresponding RGB images for the selected sample.
- Render GT 3D skeleton for the same sample to cross-check geometry.
- Allow CLI selection of config path, split, and sample index/count.

**Non-Goals:**
- No changes to dataset formats or preprocessing outputs.
- No new visualization dependencies or training-time changes.
- No automated evaluation metrics; this is a visual inspection tool only.

## Decisions

- **Use dataset config and pipeline for sample construction.**
  This ensures keypoints are produced the same way as in training. Alternative would be
  reading files directly, but that would drift from training behavior.

- **Overlay 2D keypoints per view using matplotlib.**
  This keeps dependencies minimal and is consistent with existing visualization scripts.

- **Assume H36M multiview layout with `input_rgb` shape [B,V,T,C,H,W] and
  `input_pose2d_rgb` shape [B,V,T,J,2].**
  The script will select a view index and a time index (e.g., last frame) for overlay.

## Risks / Trade-offs

- **Risk:** Configs with `load_rgb: false` cannot overlay images.
  → **Mitigation:** Detect missing RGB and print a clear error or skip overlay plots.

- **Risk:** Pipelines may produce 2D keypoints in a different key name.
  → **Mitigation:** Allow CLI override of the 2D key name (default: `input_pose2d_rgb`).
