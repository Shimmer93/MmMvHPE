## Context

H36MDataset currently yields one camera view per sample and returns a single frame’s 3D joints (optionally from a sequence). FusionFormer’s paper setting uses multi-view, multi-frame RGB inputs with 2D keypoints from an external estimator. We need an H36M multiview RGB pipeline that can provide per-camera RGB sequences and GT-derived 2D keypoints as a stand-in estimator, plus a config aligned with paper settings (T=27, B=2, 17 joints). Compute-heavy logic (Transformer fusion) stays in `models/` on GPU; dataset and GT adapters remain CPU-side.

## Goals / Non-Goals

**Goals:**
- Provide an H36M multiview RGB dataset path that returns multiple camera views per sample.
- Generate GT-derived 2D keypoints from 3D camera-space joints for each view.
- Add a FusionFormer H36M config with T=27, B=2, 17 joints, all cameras configurable.

**Non-Goals:**
- Training or integrating a real 2D estimator (ViTPose) in this change.
- Adding depth or point cloud modalities for H36M.
- Evaluating exact protocol variants beyond standard MPJPE/P-MPJPE.

## Decisions

- **New dataset class:** Create a dedicated multiview dataset (e.g., `H36MMultiViewDataset`) to avoid breaking single-view configs.
- **View stacking:** Return RGB as `[V, T, H, W, 3]` and GT 3D joints as `[T, J, 3]` in camera space; derive 2D per view using perspective projection with camera intrinsics.
- **Config-driven cameras:** The dataset uses a `cameras` list from config; defaults to all 4 cameras in the new config.
- **FusionFormer parameters:** Expose `seq_len` (T) and `num_blocks` (B) via config; set to T=27, B=2 for paper alignment.
- **Compute placement:** Dataset and GT adapter transformations remain in `datasets/` and `datasets/transforms/`; fusion stays in the existing FusionFormer aggregator.

## Risks / Trade-offs

- [Multi-view indexing correctness] → Mitigation: reuse existing H36M metadata helpers and keep camera-specific pose loading per view.
- [2D projection ambiguity] → Mitigation: document that GT-2D is derived by perspective projection using camera intrinsics.
- [Increased memory use] → Mitigation: allow configurable `seq_len` and camera list; start with all cameras only for replication runs.
