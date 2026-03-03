## 1. Metric implementation

- [x] 1.1 Add `SMPL_PCMPJPE` computation path in `metrics/smpl_metrics.py` with pelvis translation and root-orientation alignment plus explicit input validation.
- [x] 1.2 Add `PCMPJPE` computation path in `metrics/mpjpe.py` using pelvis alignment and non-SMPL orientation derivation consistent with `datasets/panoptic_preprocessed_dataset_v1.py`.
- [x] 1.3 Ensure new metrics are wired into existing metric registry/factory paths so they can be referenced directly by config names.

## 2. Config and docs integration

- [x] 2.1 Patch the requested RGB+PC-related config files in `configs/` to include `PCMPJPE` and/or `SMPL_PCMPJPE` alongside existing metrics.
- [x] 2.2 Add or update documentation under `docs/` describing metric semantics, assumptions (pelvis index/orientation convention), and example config usage.

## 3. Validation

- [x] 3.1 Add or update tests covering near-zero error cases under pure pelvis translation+rotation offsets and explicit failure cases for invalid inputs.
- [x] 3.2 Run targeted validation using `uv run` (tests and/or metric-focused command invocations) and confirm configs load with the new metric names.
