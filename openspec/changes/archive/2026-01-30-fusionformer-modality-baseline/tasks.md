## 0. Paper References (FusionFormer)

- [x] 0.1 Review FusionFormer architecture overview in Method section (paper: `/home/yzhanghe/.papers/27849-Article Text-31903-1-2-20240324 (1).pdf`, lines ~155–176 describe the 4-stage pipeline; Figure 1 caption at ~171–173)
- [x] 0.2 Review encoder input shaping + learnable positional encoding (paper lines ~181–225; V×T tokens, LN + learnable PE)
- [x] 0.3 Review unified fusion encoder-decoder blocks + shared block count B (paper lines ~231–259)
- [x] 0.4 Review 3D regression head (Conv1d temporal aggregation + 2-layer MLP) and MPJPE loss (paper lines ~289–309)
- [x] 0.5 Review decoder per-view fusion with global feature (paper lines ~319–360)

## 1. Model Scaffold

- [x] 1.1 Add FusionFormer baseline module under `models/` with encoder-decoder fusion and temporal head (paper refs: Method section lines ~155–360; 3D head lines ~289–304)
- [x] 1.2 Register the model name for config loading

## 2. GT Pose Adapters

- [x] 2.1 Add GT 2D pose adapter to supply RGB 2D keypoints to the model input
- [x] 2.2 Add GT 3D pose adapter to supply PC 3D keypoints to the model input
- [x] 2.3 Add config toggles to enable/disable GT adapters per modality

## 3. Config and Wiring

- [x] 3.1 Add a mini train/test config based on `configs/dev/humman_leir_rgbpc.yml` with `train_mini`/`test_mini`
- [x] 3.2 Wire model inputs in the training pipeline to feed GT pose adapters

## 4. Validation

- [x] 4.1 Run a short `uv run` sanity check (single batch or few steps) to verify forward pass
- [x] 4.2 Document baseline assumptions and usage in a short note (if needed under `docs/`)
