## Context

`scripts/visualize_sam3d_body_rerun.py` currently produces usable rerun traces but with multiple visualization-quality defects:

- RGB color mismatch in displayed input/overlay panels.
- 2D keypoints are plotted as points only (no edge connectivity).
- 3D keypoint connectivity does not match SAM-3D-Body keypoint definition.
- Side-view 3D keypoints and mesh are transformed differently, causing misalignment.
- GT visualization is absent, reducing qualitative comparison value.

This impacts the benchmark/demo workflow because rerun is used as the visual sanity layer for cross-method comparisons. The change is isolated to script/helper visualization code and docs; it does not touch `main.py` training/evaluation flow or model learning code.

## Goals / Non-Goals

**Goals:**
- Correct RGB display color handling for SAM-3D-Body rerun output.
- Add topology-aware 2D/3D skeleton connectivity using the SAM-3D-Body joint convention.
- Make front/side transforms consistent between mesh and keypoints.
- Add GT overlays/logs (keypoints, and mesh where recoverable) in shared namespaces.
- Preserve existing CLI ergonomics, especially explicit `--render-mode` semantics.

**Non-Goals:**
- No changes to SAM-3D-Body model weights or inference internals.
- No new quantitative metrics or evaluator changes.
- No refactor of the broader training pipeline (`models/model_api.py`, `main.py`).

## Decisions

1. Add explicit topology mapping for SAM-3D-Body keypoints in visualization layer.
Why:
- We need deterministic, inspectable edge definitions independent of renderer defaults.
Design:
- Define a SAM-3D-Body skeleton edge list in `scripts/visualize_sam3d_body_rerun.py` or `scripts/rerun_utils/logging3d.py` and use it for 2D/3D line drawing.
Alternative:
- infer edges from existing MMHPE skeleton classes.
Rejected because SAM-3D-Body joint order may differ.

2. Use one shared transform path for mesh and keypoints before front/side logging.
Why:
- Misalignment indicates different coordinate transforms are applied.
Design:
- Normalize orientation once (including optional 180deg Y correction), then derive side view from the exact same transformed tensor for both mesh and keypoints.
Alternative:
- keep separate mesh/keypoint transform code blocks.
Rejected because it is error-prone and caused current drift.

3. Treat image handling as explicit dtype/range/channel contract.
Why:
- The current artifact (few discrete colors + mostly black image) indicates possible value-range collapse or dtype quantization in addition to channel-order issues.
Design:
- At each stage (dataset sample -> preprocessing helper -> overlay draw -> rerun log), enforce and validate:
  - shape `(H, W, 3)`,
  - channel order contract,
  - numeric range contract (`uint8` in `[0,255]` or float in `[0,1]`),
  - dtype conversion points.
- Keep internal image arrays in one canonical format for rerun logging, and only convert when required by drawing backend.
- Add debug logs/guards for min/max/dtype before `rr.Image` logging to detect collapsed dynamic range.
Alternative:
- ad hoc channel flips at call sites.
Rejected due to fragility.

4. Add GT logging opportunistically based on available sample fields.
Why:
- Different dataset configs expose different GT fields.
Design:
- If `gt_keypoints` exists, log GT skeleton.
- If SMPL params exist and can be converted, log GT mesh.
- If GT mesh is unavailable, still log GT keypoints and record metadata in `world/info/*`.
Alternative:
- require GT mesh always.
Rejected because it would break valid configs lacking full SMPL targets.

5. Keep fixes inside scripts + rerun helpers only.
Why:
- Low-risk, incremental change that does not impact core model execution.
Design:
- Code changes confined to `scripts/visualize_sam3d_body_rerun.py`, `scripts/rerun_utils/*.py`, and docs.
Alternative:
- introducing new framework abstraction layer.
Rejected as unnecessary churn for this scope.

## Risks / Trade-offs

- [Incorrect SAM-3D-Body joint order assumption] → Mitigation: verify upstream keypoint definition from `third_party/sam-3d-body` source and annotate mapping in docs.
- [Different datasets expose GT in different shapes/frames] → Mitigation: add strict shape checks + clear warnings when GT cannot be rendered.
- [Color fix regresses non-SAM scripts] → Mitigation: keep SAM-specific image handling in SAM script path; avoid changing MMHPE script behavior unless explicitly required.
- [Image still appears quantized/black after channel fix] → Mitigation: add staged min/max/dtype assertions and save intermediate debug images during validation.
- [Render-mode behavior ambiguity] → Mitigation: keep current explicit mode contract and document per-mode output expectations.

## Migration Plan

1. Verify SAM-3D-Body keypoint definition/order and encode topology mapping used for 2D/3D connectivity.
2. Refactor SAM rerun script to apply consistent coordinate transforms for mesh/keypoints and fix RGB/BGR handling.
3. Add GT visualization path with field-detection logic and robust fallback behavior.
4. Update `docs/rerun_visualization.md` with topology source, color-space notes, and render-mode outputs.
5. Run smoke checks for `overlay`, `auto`, and `mesh` modes; save `.rrd` outputs under `logs/`.

## Open Questions

- None for artifact stage; implementation should confirm the exact SAM-3D-Body joint topology from upstream code and record source file path in docs.
