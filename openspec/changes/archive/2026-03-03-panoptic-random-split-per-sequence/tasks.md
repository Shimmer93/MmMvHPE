## 1. Split Config Contract Updates

- [x] 1.1 Update `configs/datasets/panoptic_split_config.yml` to add a new `temporal_split` section (`ratio`, `random_seed`, train/val entries) and add inline comments defining split semantics.
- [x] 1.2 Populate `cross_subject_split.train_dataset.sequences` and `cross_subject_split.val_dataset.sequences` with the current sequence-level train/val partition so legacy sequence-level behavior is explicit.
- [x] 1.3 Update Panoptic experiment/visualization configs that require per-sequence temporal head/tail splitting to use `split_to_use: temporal_split`.

## 2. Dataset Loader Implementation

- [x] 2.1 Modify `datasets/panoptic_preprocessed_dataset_v1.py` `_resolve_split_selection()` to support `split_to_use: temporal_split` without sequence permutation assignment.
- [x] 2.2 Implement deterministic per-sequence temporal cutoff logic on synchronized `frame_ids` (head for train, tail for val/test) before sample window generation.
- [x] 2.3 Ensure temporal split is applied on common synchronized frame IDs so RGB/depth/lidar sample alignment is preserved.
- [x] 2.4 Add fail-fast validation for invalid `temporal_split.ratio` and empty per-sequence partition results under `strict_validation=True`.

## 3. Validation and Regression Checks

- [x] 3.1 Run a deterministic split check with `uv run python` to verify repeated initialization yields identical per-sequence frame membership for `temporal_split`.
- [x] 3.2 Run a dataset instantiation check with `uv run python` for both `split_to_use: temporal_split` and `split_to_use: cross_subject_split` to confirm non-empty splits and expected sequence membership.
- [x] 3.3 Run `uv run python scripts/visualize_inference_rerun.py ...` with a Panoptic config using `temporal_split` to verify sequence allowlist intersection remains valid for train/val workflows.

## 4. Documentation Updates

- [x] 4.1 Update `docs/panoptic_kinoptic_preprocess.md` with a “Split Semantics” section documenting `temporal_split` vs `cross_subject_split` and migration guidance from legacy `random_split` usage.
- [x] 4.2 Update `docs/README.md` to reference the Panoptic split semantics documentation entry for discoverability.
- [x] 4.3 Add concrete command examples (using `uv run`) showing how to select `temporal_split` and explicit `cross_subject_split` in Panoptic configs.
