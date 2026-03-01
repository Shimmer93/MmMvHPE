## 1. Dataset Toggle

- [x] 1.1 Add `load_rgb` (or similar) flag to H36MMultiViewDataset init/config and thread it through dataset creation
- [x] 1.2 Gate RGB frame loading in H36MMultiViewDataset when the flag is disabled and return `input_rgb` as None or omitted

## 2. Pipeline/Config Updates

- [x] 2.1 Update H36M configs to demonstrate keypoints-only training with RGB loading disabled
- [x] 2.2 Ensure H36M pipelines avoid RGB-dependent transforms when RGB loading is disabled (document or adjust as needed)

## 3. Verification

- [x] 3.1 Run a small H36M sanity check with RGB loading disabled using `uv run` and confirm training/validation runs
