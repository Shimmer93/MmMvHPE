## Context

`scripts/visualize_sam3d_body_rerun.py` currently logs one frame per selected sample window. This limits temporal inspection of prediction stability and frame-to-frame failures. The script already has timeline support (`set_frame_timeline`) and sequence-window inputs, so extending to multi-frame playback is a focused script-level change.

## Goals / Non-Goals

**Goals:**
- Add explicit `--num-frames` control to visualize multiple frames from the selected sample window.
- Keep backward compatibility for existing single-frame behavior.
- Define deterministic frame selection when combined with `--frame-index`.
- Keep output namespace and render-mode behavior unchanged.
- Ensure multi-frame execution works for `overlay`, `auto`, and `mesh` modes.

**Non-Goals:**
- No dataset class refactor.
- No model/training/evaluation changes in `main.py` or `models/`.
- No new metrics; visualization-only behavior.

## Decisions

1. Add `--num-frames` with default `1`.
Why:
- Preserves current CLI behavior while enabling temporal mode.
Design:
- `--num-frames 1` keeps current single-frame path.
- `--num-frames N>1` iterates selected frame indices and logs one timeline step per frame.
Alternative:
- replace `--frame-index` entirely.
Rejected due to backward-compatibility cost.

2. Keep `--frame-index` as anchor for multi-frame selection.
Why:
- Users need control over where temporal extraction starts/centers.
Design:
- If `frame-index >= 0`: use it as start index and take up to `num_frames` contiguous frames.
- If `frame-index == -1`: center window around sequence midpoint.
- Clip to valid `[0, T-1]` bounds.
Alternative:
- always start from 0.
Rejected because it reduces debug control.

3. Run SAM-3D-Body inference per frame in loop and reuse existing logging helpers.
Why:
- Minimal change; avoids new batching assumptions in third-party estimator.
Design:
- Per frame: log input, run estimator, log overlay/pred/GT, set timeline.
Alternative:
- batch frames and infer once.
Deferred; estimator API is image-centric.

4. Timeline key uses local frame offset, metadata stores absolute index.
Why:
- Rerun playback should be contiguous even when source frame indices are sparse/clipped.
Design:
- `set_frame_timeline(local_idx, sample_id=...)`
- log `world/info/source_frame_index` as original frame index.
Alternative:
- timeline equals source frame index.
Rejected because playback can appear discontinuous.

## Risks / Trade-offs

- [Longer runtime for mesh mode over many frames] → Mitigation: keep default `num_frames=1` and document expected cost.
- [Ambiguity around frame-index semantics] → Mitigation: document exact selection rules and add explicit metadata logging.
- [Edge clipping at sequence boundaries] → Mitigation: deterministic clipping and clear logging of selected indices.
- [Potential mismatch across views if only center view is inferred] → Mitigation: keep current center-view inference contract unchanged; document scope.

## Migration Plan

1. Add CLI and frame-selection utility in `scripts/visualize_sam3d_body_rerun.py`.
2. Refactor single-frame logic into frame-loop while preserving existing render outputs.
3. Add timeline and frame-index metadata logging.
4. Update `docs/rerun_visualization.md` with `--num-frames` usage and semantics.
5. Run smoke validation with `num_frames=1` and `num_frames>1` for at least overlay + mesh.

## Open Questions

- Should multi-frame mode support stride (`frame-step`) now, or keep contiguous frames only? Current proposal keeps contiguous only to minimize scope.
