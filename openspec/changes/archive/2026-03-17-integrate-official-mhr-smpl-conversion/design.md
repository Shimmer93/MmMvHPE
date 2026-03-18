## Context

MMHPE currently evaluates and visualizes SAM3D on HuMMan by taking raw SAM/MHR outputs and applying a local heuristic MHR70-to-SMPL24 joint remap. That approach avoids extra dependencies, but it is not a true SMPL reconstruction path and produces visibly incorrect torso and shoulder topology in qualitative checks. The official upstream MHR conversion code provides a more correct path: fit SMPL to MHR vertices produced by SAM3D, then evaluate and visualize the fitted SMPL outputs.

This change touches multiple parts of the current architecture:
- `scripts/run_sam3d_eval_suite.py` for HuMMan evaluation
- `scripts/visualize_sam3d_humman_smpl24_conversion.py` and related SAM3D visualization/export utilities
- shared SAM3D helper code under `misc/`
- environment/dependency expectations for SAM3D tooling
- documentation under `docs/`

Constraints:
- keep the integration incremental and avoid changing `main.py`, `datasets/data_api.py`, or `models/model_api.py`
- do not expand scope into Panoptic conversion in the first pass
- fail fast when official conversion dependencies or model assets are missing
- keep output paths under `logs/` deterministic and compatible with existing script-driven workflows

## Goals / Non-Goals

**Goals:**
- Replace the heuristic HuMMan MHR70-to-SMPL24 remap with the official MHR-to-SMPL conversion path for SAM3D evaluation and visualization.
- Preserve the current script-driven workflow: users should still run `uv run python scripts/...` with config-selected datasets and outputs under `logs/`.
- Centralize the conversion logic in one wrapper module so evaluation and visualization use the same implementation.
- Validate the environment explicitly before running conversion, including MHR package availability and SMPL model availability.
- Keep the existing raw SAM/MHR outputs available for side-by-side comparison in visualizations.

**Non-Goals:**
- Converting Panoptic SAM3D evaluation to official MHR-to-SMPL in this change.
- Integrating official conversion into training-time model code or Lightning modules.
- Reworking the generic dataset or model APIs.
- Supporting multiple approximate fallback conversion modes when official dependencies are missing.

## Decisions

### 1. Use a dedicated wrapper module for official conversion
Create a shared helper under `misc/` that owns:
- environment validation for `mhr`, `smplx`, and required model files
- initialization of `MHR` and `Conversion`
- conversion of SAM3D output dictionaries to fitted SMPL outputs
- extraction of SMPL24 joints and optional meshes/parameters

Rationale:
- evaluation and visualization currently diverged because they each embedded their own conversion assumptions
- one wrapper gives a single contract and a single place to enforce fail-fast behavior

Alternative considered:
- call the upstream conversion scripts directly from each consumer
- rejected because it would duplicate initialization, make batched use harder, and fragment error handling

### 2. HuMMan-only replacement in the first phase
Apply the official conversion only to HuMMan scripts and leave Panoptic conversion untouched.

Rationale:
- the current correctness issue was discovered on HuMMan SMPL24 visualization
- HuMMan GT is already SMPL24-aligned, so it is the cleanest target for the official converter
- Panoptic currently evaluates against Panoptic joints, so switching Panoptic would expand scope into a different adapter problem

Alternative considered:
- replace both HuMMan and Panoptic conversion paths immediately
- rejected because it would mix two different representation problems into one change

### 3. Batch conversion inside evaluation, frame conversion inside visualization
For `scripts/run_sam3d_eval_suite.py`, collect SAM3D outputs for a chunk of HuMMan frames and run the official conversion in batches. For visualization scripts, convert only the requested frames or samples.

Rationale:
- the upstream converter already supports batched processing and SMPL fitting can be expensive
- evaluation throughput matters more than single-sample visualization latency
- visualization should stay simple and deterministic per requested sample

Alternative considered:
- convert each HuMMan frame independently in all scripts
- rejected because it would add unnecessary overhead to evaluation sweeps

### 4. Treat missing official dependencies as a hard error
If `mhr`, `smplx`, or required SMPL assets are missing, scripts should raise explicit errors instead of silently falling back to the heuristic adapter.

Rationale:
- silent fallback would make results incomparable and obscure whether the official path was actually used
- this change is specifically about replacing an incorrect approximation with the correct path

Alternative considered:
- keep heuristic mapping as an automatic fallback
- rejected because it undermines the purpose of the change and makes outputs ambiguous

### 5. Keep raw MHR outputs visible in visualization artifacts
The HuMMan visualization path should continue to export raw SAM/MHR overlays or 3D plots alongside converted SMPL outputs and GT SMPL24.

Rationale:
- raw-vs-converted comparison is useful for validating the converter itself
- it preserves the current debugging value of the conversion-check script

Alternative considered:
- show only GT and converted SMPL outputs
- rejected because it would hide the difference between SAM output quality and conversion quality

### 6. Document dependency setup in `docs/` and, if needed, `pyproject.toml`
The final implementation should document how to obtain the official MHR package/assets. If new installable dependencies are required in the repo environment, update `pyproject.toml` without touching `uv.lock` manually.

Rationale:
- the current environment already has `smplx` and `trimesh`, but `mhr` is missing
- users need a clear setup path before the scripts can run

Alternative considered:
- rely on ad hoc local installs without repository documentation
- rejected because this change modifies the supported SAM3D evaluation workflow

## Risks / Trade-offs

- [Extra dependency surface from official MHR conversion] → Keep the integration isolated in helper modules and document the setup explicitly; fail fast when unavailable.
- [Higher runtime cost from SMPL fitting] → Use batched conversion in evaluation and keep single-sample conversion for visualization only.
- [Backward incompatibility with earlier HuMMan SAM3D metrics and visual outputs] → Document that outputs are intentionally not directly comparable to heuristic-adapter runs.
- [Upstream converter assumptions may not exactly match our SAM3D checkout] → Validate against our existing `SAM3DBodyEstimator.process_one_image(...)` output structure before replacing the current path fully.
- [Potential CUDA/CPU constraints in the official converter] → Expose device and batch-size control in the wrapper and document expected performance limits.
