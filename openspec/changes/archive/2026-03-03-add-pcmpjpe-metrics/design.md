## Context

MMHPE evaluation currently reports MPJPE-family metrics for both SMPL outputs and direct keypoint outputs. In practical RGB+PC runs, model prediction and GT can differ in global pelvis translation and root orientation even when articulated pose quality is good. The change needs to add a pelvis-centered metric variant consistently across `metrics/smpl_metrics.py` and `metrics/mpjpe.py`, while preserving existing metric behavior and config compatibility.

Constraints:
- Keep current config-driven execution (`main.py` + YAML configs) unchanged except for optional new metric names.
- Fail fast on invalid tensor shapes or missing required joints/rotations.
- Reuse existing coordinate/orientation conventions already used in `datasets/panoptic_preprocessed_dataset_v1.py` for non-SMPL root-rotation removal.

## Goals / Non-Goals

**Goals:**
- Add `SMPL_PCMPJPE` and `PCMPJPE` with deterministic pelvis translation + orientation alignment semantics.
- Keep implementation incremental in existing metric modules without refactoring the broader evaluation pipeline.
- Allow enabling new metrics from configs used in the current RGB+PC pipeline.
- Add tests that validate alignment behavior and basic integration.

**Non-Goals:**
- Changing dataset preprocessing logic or coordinate conventions globally.
- Adding new dependencies or changing CUDA/runtime stack.
- Replacing or removing existing MPJPE metrics.

## Decisions

1. Implement metrics in place in existing modules.
- Decision: extend `metrics/smpl_metrics.py` with `SMPL_PCMPJPE` and `metrics/mpjpe.py` with `PCMPJPE`.
- Rationale: keeps backward compatibility and follows current metric registration/usage patterns.
- Alternative considered: new shared pelvis-alignment utility module. Rejected for now to avoid unnecessary refactor scope.

2. Use explicit pelvis-center and pelvis-orientation normalization before MPJPE.
- Decision: subtract pelvis position offset (prediction vs GT), then rotate prediction into GT pelvis frame before computing joint distance.
- Rationale: directly matches requested metric definition and isolates articulated error from global root mismatch.
- Alternative considered: only translation-centering (classic P-MPJPE-like variant). Rejected because orientation alignment is a stated requirement.

3. Non-SMPL orientation estimation follows dataset convention.
- Decision: compute non-SMPL root orientation using the same convention as `remove_root_rotation` in `datasets/panoptic_preprocessed_dataset_v1.py`.
- Rationale: avoids silent frame-convention mismatches between preprocessing and metric-time alignment.
- Alternative considered: infer orientation from arbitrary torso vectors. Rejected because it may diverge from established project convention.

4. Config rollout is additive.
- Decision: patch requested configs by appending new metric names instead of replacing existing metrics.
- Rationale: keeps historical metric continuity and avoids breaking existing dashboards/scripts.

5. Documentation update in docs/.
- Decision: add/update a docs file describing semantics and config usage examples for new metrics.
- Rationale: project convention prefers docs/ for behavior and usage details over dense inline comments.

## Risks / Trade-offs

- [Risk] Non-SMPL orientation convention mismatch across datasets → Mitigation: reuse the exact existing `remove_root_rotation` logic path and add shape/assumption checks.
- [Risk] Pelvis joint index assumptions may vary by keypoint format → Mitigation: use existing keypoint metadata/index mapping where available, raise explicit errors otherwise.
- [Risk] Additional metric computation cost in evaluation loops → Mitigation: keep operations vectorized and limited to already available tensors.
- [Risk] Config changes may miss some active experiment files → Mitigation: patch the configs explicitly requested and validate by running targeted config-based tests/commands.
