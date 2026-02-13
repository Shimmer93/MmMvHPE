## 1. Shared Rerun Core Extraction

- [x] 1.1 Create shared rerun helper module(s) under `scripts/` (for example `scripts/rerun_utils/`) for config-driven input layout parsing, timeline stepping, and common rerun logging paths.
- [x] 1.2 Move reusable logic from `scripts/visualize_inference_rerun.py` into the shared module(s) without changing baseline behavior.
- [x] 1.3 Add strict validation/errors in shared layout parsing for missing modality lists or inconsistent per-modality view counts.

## 2. MMHPE Visualization Script Migration

- [x] 2.1 Update `scripts/visualize_inference_rerun.py` to consume shared helper APIs for input panel/entity creation.
- [x] 2.2 Ensure `scripts/visualize_inference_rerun.py` keeps existing CLI arguments and produces standardized namespaces (`world/inputs/*`, `world/front/*`, `world/side/*`, `world/info/*`).
- [x] 2.3 Run a smoke test with an existing MMHPE config/checkpoint and verify rerun log output is generated under `logs/`.

## 3. SAM-3D-Body Rerun Script

- [x] 3.1 Add a SAM-3D-Body rerun script under `scripts/` that loads dataset config, selects sample by split/index, and runs inference.
- [x] 3.2 Implement explicit CLI `--render-mode` with `mesh|overlay|auto` and enforce forced-mode behavior.
- [x] 3.3 Enforce checkpoint contract at `/opt/data/SAM_3dbody_checkpoints/` for `model_config.yaml`, `model.ckpt`, and `mhr_model.pt` with fail-fast actionable errors.
- [x] 3.4 Wire SAM-3D-Body outputs through the shared rerun core using the same namespace conventions as MMHPE visualization.

## 4. Validation and Documentation

- [x] 4.1 Validate MMHPE visualization path with a runnable command (for example `uv run python scripts/visualize_inference_rerun.py ...`) and confirm config-driven modality/view layout in rerun.
- [x] 4.2 Validate SAM-3D-Body path with runnable commands for each render mode (for example `uv run python scripts/<sam3d_rerun_script>.py --render-mode overlay ...`).
- [x] 4.3 Add/update docs in `docs/` covering script purpose, CLI options, expected rerun output structure, and command examples.
- [x] 4.4 Record known limitations and debugging notes for rendering backends and dataset/view assumptions in `docs/`.
