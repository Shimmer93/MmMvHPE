## Context

Current rerun scripts (`scripts/visualize_inference_rerun.py`, `scripts/visualize_sam3d_body_rerun.py`) expose temporal CLI controls, but behavior can diverge from config-defined model input contracts. In MMHPE, model input windows are defined by dataset + pipeline config (`configs/*`, dataset classes under `datasets/`), while visualization users expect CLI flags to control what is shown in rerun timelines. The immediate target is demo visualization reliability (HuMMan inference and SAM3D scripts), with current demo configs using `seq_len: 1`.

The design must avoid changing prediction semantics in `models/model_api.py` or training behavior in `main.py`; this change is script/data-flow alignment for visualization.

## Goals / Non-Goals

**Goals:**
- Keep model feeding logic strictly config-driven (dataset split, modality/view selection, `seq_len`, pipeline).
- Make visualization frame selection explicit and script-consistent via CLI (`sample-idx`, `frame-index`, `num-frames`).
- Ensure both inference and SAM rerun scripts produce comparable timelines for the same sample/window settings.
- Consume temporal GT fields when available (`gt_keypoints_seq`, `gt_smpl_params_seq`) and gracefully fallback to single-frame GT fields.
- Improve rerun metadata for reproducibility (`sample_id`, `source_frame_index`, `num_visualized_frames`, coordinate mode).

**Non-Goals:**
- No architectural/model change to force sequence-to-sequence outputs.
- No loss/metric/training loop changes in `models/` or `main.py`.
- No new dependencies or CUDA stack changes.
- No change to checkpoint format.

## Decisions

1) **Separate model input construction from visualization sampling**
- Decision: The scripts always fetch one dataset sample using config-driven dataset construction (`create_dataset(...)` from config-selected split/pipeline), then apply CLI frame controls only when logging timeline entries.
- Rationale: Preserves model contract and avoids accidental mismatch between config `seq_len` and visualized frames.
- Alternative considered: Override dataset `seq_len` from CLI. Rejected because it mutates model input semantics and breaks reproducibility against config/checkpoint assumptions.

2) **Common temporal selection semantics across scripts**
- Decision: Align frame index behavior with SAM script semantics:
  - `frame-index >= 0`: start at explicit frame
  - `frame-index == -1`: centered window start
  - `num-frames`: contiguous frame count clipped to available sequence length
- For `seq_len: 1`, interpret `num-frames` as timeline length across consecutive samples/windows starting from `sample-idx` (instead of within-window frames), so motion can still be visualized.
- Rationale: Eliminates confusing differences between inference and SAM visualizations.
- Alternative considered: Keep script-specific policies. Rejected due to persistent operator confusion and non-comparable timelines.

3) **Temporal GT consumption with fallback**
- Decision: Prefer `gt_keypoints_seq` / `gt_smpl_params_seq`; fallback to `gt_keypoints` / `gt_smpl_params` when temporal labels are unavailable.
- Rationale: Supports both temporal-capable and legacy samples/configs.
- Alternative considered: Require temporal GT always. Rejected to preserve backward compatibility and avoid hard failures for existing datasets.

4) **Dataset support for temporal SMPL GT (visualization-only enabler)**
- Decision: Add optional dataset output `gt_smpl_params_seq` gated by config flag (`return_smpl_sequence`) in HuMMan preprocessed datasets.
- Rationale: Needed for dynamic GT mesh rendering without affecting model training contract.
- Alternative considered: Reconstruct temporal SMPL from raw files directly in scripts. Rejected because dataset already owns sample assembly and camera/world transformations.

5) **Config-first behavior with current `seq_len: 1` demos**
- Decision: Keep demo configs at `seq_len: 1` for current models; temporal CLI options remain valid but effectively produce one frame in those configs.
- Rationale: Matches actual model/runtime constraints while keeping interface ready for future temporal configs.
- Alternative considered: Force higher `seq_len` in demo configs. Rejected because it can violate current checkpoint/model assumptions.

## Risks / Trade-offs

- [Risk] Ambiguity of `num-frames` meaning between `seq_len=1` and `seq_len>1`.  
  -> Mitigation: Define explicit behavior: for `seq_len>1` use within-window temporal frames; for `seq_len=1` advance across consecutive samples/windows.

- [Risk] Additional temporal GT tensors increase sample memory and I/O.  
  -> Mitigation: Keep `return_smpl_sequence` optional and enabled only in visualization-focused configs.

- [Risk] View/modality count inconsistencies in config can still raise runtime layout errors.  
  -> Mitigation: Keep strict validation in rerun layout utilities and provide explicit error messages.

- [Risk] Different coordinate-space settings can make motion appear inconsistent across panels.  
  -> Mitigation: Log coordinate mode/reference sensor metadata and keep transform path explicit in scripts.

## Migration Plan

1. Update HuMMan dataset classes to optionally emit temporal SMPL GT (`gt_smpl_params_seq`).
2. Update both rerun scripts to consume temporal GT + consistent frame-selection semantics, including cross-sample timeline mode when `seq_len=1`.
3. Keep or set demo configs to `seq_len: 1` for current models; enable temporal GT flags where needed.
4. Add docs note under `docs/` describing:
   - config-driven model input vs CLI-driven visualization frame selection,
   - expected behavior for `seq_len: 1` vs `seq_len > 1`,
   - reproducibility metadata in rerun logs.
5. Validate by generating small `.rrd` smoke outputs for both scripts and checking frame metadata continuity.

Rollback:
- Revert script-level frame-selection changes and disable `return_smpl_sequence` in configs.
- Dataset fallback ensures legacy single-frame GT path remains available.

## Open Questions

- Do we want a shared helper module for frame-index policy to prevent future drift between rerun scripts?
