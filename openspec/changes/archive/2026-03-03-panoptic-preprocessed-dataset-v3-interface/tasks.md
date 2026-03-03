## 1. Dataset Class Implementation

- [x] 1.1 Add `datasets/panoptic_preprocessed_dataset_v1.py` implementing Panoptic preprocessed loading from `/opt/data/panoptic_kinoptic_single_actor_cropped` by default
- [x] 1.2 Implement strict sequence artifact validation at index build time with explicit error messages (hard-fail default)
- [x] 1.3 Implement sample construction to match `humman_dataset_v3` training-facing field contract, including RGB/depth/GT/calibration-related fields
- [x] 1.4 Normalize Panoptic camera naming/metadata to the style consumed by existing transforms and model paths

## 2. Split and Sequence Selection

- [x] 2.1 Add split config parsing compatible with existing dataset split workflow (`split_config`, split name selection)
- [x] 2.2 Implement deterministic sequence-level ratio split fallback (stable ordering + fixed seed behavior) when explicit split lists are absent
- [x] 2.3 Implement sequence allowlist filtering and intersection with split-selected sequences
- [x] 2.4 Enforce hard-fail default for unknown/missing sequences referenced by split config or sequence filters

## 3. Dataset API and Config Wiring

- [x] 3.1 Register the new Panoptic dataset class in `datasets/data_api.py`
- [x] 3.2 Add/adjust Panoptic dataset config entries under `configs/datasets/` for train/val/test usage
- [x] 3.3 Ensure config defaults and examples keep existing HuMMan workflows unchanged

## 4. Validation

- [x] 4.1 Run a dataset instantiation smoke test with `uv run python` using `161029_piano2` to verify indexing and split selection
- [x] 4.2 Run a minimal dataloader/sample retrieval check with `uv run python` on `161029_piano2` and assert required `humman_dataset_v3`-compatible fields exist
- [x] 4.3 Run a deterministic split reproducibility check (same seed/config yields identical sequence membership across repeated runs)

## 5. Documentation

- [x] 5.1 Add/update docs in `docs/` describing Panoptic preprocessed dataset contract, expected folder layout, and strict validation behavior
- [x] 5.2 Add command examples for split-config-driven usage and deterministic ratio split fallback
