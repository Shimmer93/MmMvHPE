## 1. Environment and dependency setup

- [x] 1.1 Inspect the official MHR conversion package requirements and decide the minimal repo integration path for `mhr` and related assets without changing unrelated MMHPE workflows.
- [x] 1.2 Update environment-facing code and, if required, `pyproject.toml` so official HuMMan conversion dependencies are documented and discoverable without editing `uv.lock` manually.
- [x] 1.3 Add fail-fast preflight checks in a shared helper module under `misc/` for `mhr`, `smplx`, and required SMPL model assets used by the official conversion path.

## 2. Shared official conversion wrapper

- [x] 2.1 Add a shared wrapper module under `misc/` that initializes the official MHR `Conversion` flow and exposes a stable MMHPE-facing API for converting SAM3D outputs to fitted SMPL outputs.
- [x] 2.2 Implement SMPL24 joint extraction and optional mesh/parameter return values in the wrapper so HuMMan scripts can consume one common output contract.
- [x] 2.3 Add explicit validation for required SAM3D output fields (`pred_vertices`, `pred_cam_t`, or other accepted official-converter inputs) and fail with actionable errors when inputs are incomplete.
- [x] 2.4 Keep the existing heuristic HuMMan adapter isolated from the new wrapper so the replacement is explicit and easy to verify.

## 3. HuMMan evaluation integration

- [x] 3.1 Patch `scripts/run_sam3d_eval_suite.py` so the HuMMan path uses the official conversion wrapper instead of the local heuristic MHR70-to-SMPL24 adapter.
- [x] 3.2 Add batched HuMMan conversion inside `scripts/run_sam3d_eval_suite.py` so multiple SAM3D outputs can be converted efficiently during evaluation runs.
- [x] 3.3 Preserve current config-driven HuMMan split/camera selection and ensure GT and converted predictions are compared in the same RGB camera coordinate frame.
- [x] 3.4 Keep Panoptic evaluation behavior unchanged in `scripts/run_sam3d_eval_suite.py`.

## 4. HuMMan visualization integration

- [x] 4.1 Patch `scripts/visualize_sam3d_humman_smpl24_conversion.py` to replace the heuristic adapter with the official conversion wrapper.
- [x] 4.2 Ensure the visualization export continues to include all three views: GT SMPL24, raw SAM/MHR output, and official converted SMPL output.
- [x] 4.3 If needed, patch related SAM3D visualization helpers so HuMMan comparison exports share the same official conversion code path instead of duplicating conversion logic.
- [x] 4.4 Ensure visualization commands fail explicitly when official conversion dependencies are unavailable instead of silently falling back.

## 5. Documentation and validation

- [x] 5.1 Add or update `docs/` documentation covering official MHR conversion setup, required assets, expected outputs under `logs/`, and example HuMMan commands.
- [x] 5.2 Run syntax validation for all modified Python files with `uv run python -m py_compile ...`.
- [x] 5.3 Run a HuMMan evaluation smoke test with `uv run python scripts/run_sam3d_eval_suite.py --scenarios humman_cross_camera_test --max-pairs-per-scenario 1 --max-frames-per-pair 1 ...` and confirm the official conversion path is used.
- [x] 5.4 Run a HuMMan visualization smoke test with `uv run python scripts/visualize_sam3d_humman_smpl24_conversion.py ...` and confirm the exported comparison artifacts are written under `logs/`.
- [x] 5.5 Compare one HuMMan sample visually to confirm the official converted SMPL topology fixes the torso/shoulder structure issue seen with the heuristic adapter.
