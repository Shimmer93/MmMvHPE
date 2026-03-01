## Context

`scripts/visualize_inference_rerun.py` has grown into a mixed script that handles dataset loading, model inference, input logging, and 3D output rendering in one file. It currently hardcodes parts of the input display layout (modality/view assumptions), which does not consistently reflect per-config modality/view settings.

At the same time, we need a SAM-3D-Body visualization path that runs on MMHPE datasets and writes rerun logs in a comparable format for demos/benchmark inspections. We already have a working SAM-3D-Body environment and checkpoint contract, so this change should focus on visualization architecture and config-driven behavior, not environment setup.

## Goals / Non-Goals

**Goals:**
- Introduce a shared rerun visualization pipeline that:
  - loads dataset/config context from config files,
  - constructs input modality/view layout from dataset config,
  - delegates inference to model-specific adapters.
- Keep `scripts/visualize_inference_rerun.py` usable while moving common logic into reusable helpers.
- Add a SAM-3D-Body adapter/script that runs inference on HuMMan RGB data and logs outputs with the same timeline semantics and path conventions.
- Standardize rerun output paths and metadata under `logs/`.

**Non-Goals:**
- No changes to model training/evaluation pipelines in `main.py`, `models/model_api.py`, or optimizer/loss logic.
- No new benchmark metrics in this change.
- No full unification of all visualization scripts in one pass (incremental refactor only).

## Decisions

1. Split visualization into shared core + model adapters.
Why:
- avoids duplicating rerun logging/layout logic across scripts,
- keeps model-specific inference code isolated.
Alternative:
- keep separate standalone scripts per model without shared code.
Rejected due to duplication and maintenance drift.

2. Config-driven input panel layout is mandatory.
Why:
- dataset config is the source of truth for modalities and per-modality view counts.
Design:
- read `dataset_cfg['params']['modality_names']` and `*_cameras_per_sample`,
- build rerun left panel and entity paths per modality/view.
Alternative:
- fixed RGB/depth/lidar panels.
Rejected because it breaks for variable multi-view/multimodal settings.

3. Preserve backward compatibility of `visualize_inference_rerun.py` CLI where practical.
Why:
- current users depend on this script for MMHPE model inspection.
Design:
- refactor internal functions first, keep CLI flags stable,
- add SAM-3D-Body script with parallel CLI style for consistency.
Alternative:
- replace old script entirely.
Rejected to reduce migration risk.

4. Use explicit rerun entity namespaces shared across adapters.
Why:
- enables consistent side-by-side comparison and simpler downstream tooling.
Design:
- inputs: `/world/inputs/<modality>/view_<i>`
- outputs: `/world/front/*`, `/world/side/*`
- metadata: `/world/info/*`
Alternative:
- adapter-specific arbitrary paths.
Rejected because it complicates comparison and debugging.

5. Add explicit rendering mode CLI switch for SAM-3D-Body visualization.
Why:
- users need deterministic behavior across machines with different EGL/GPU graphics capabilities.
Design:
- support explicit mode selection (for example `--render-mode mesh|overlay|auto`),
- `auto` can preserve current fallback behavior, while `mesh` and `overlay` force behavior.
Alternative:
- implicit fallback only.
Rejected because it obscures reproducibility during demos/benchmarks.

6. Use two entry scripts now, not one unified backend switch.
Why:
- lower migration risk and clearer ownership of model-specific arguments,
- existing `visualize_inference_rerun.py` users are not forced into a new backend abstraction immediately.
Design:
- keep MMHPE and SAM-3D-Body as separate entry scripts sharing a common internal rerun module.
Alternative:
- one script with `--backend {mmmvhpe,sam3d}`.
Deferred until shared core stabilizes and argument schemas converge.

## Risks / Trade-offs

- [Refactor introduces regressions in existing rerun script] -> Mitigation: keep old CLI contract, add small smoke run checks with existing config.
- [SAM-3D-Body mesh render backend issues (EGL/pyrender)] -> Mitigation: maintain fallback visualization mode (2D overlays) while preserving inference output logging.
- [Config variability across datasets causes shape/view edge cases] -> Mitigation: centralize input-layout parsing and sample view splitting in shared utility with strict validation/errors.
- [Over-refactor slows delivery] -> Mitigation: incremental extraction of shared logic; avoid broad framework churn.

## Migration Plan

1. Extract shared rerun helpers from `scripts/visualize_inference_rerun.py` into reusable utility module(s) under `scripts/` (for example `scripts/rerun_utils/`) for layout, modality logging, timelines, and common transforms.
2. Update `scripts/visualize_inference_rerun.py` to consume shared helpers while preserving current CLI behavior.
3. Add SAM-3D-Body rerun inference script that:
   - loads dataset via config,
   - selects RGB frames,
   - runs SAM-3D-Body inference,
   - logs results via shared rerun helpers.
4. Add docs in `docs/` with command examples for both scripts and expected outputs under `logs/`.
5. Run smoke validation for:
   - existing MMHPE rerun script on known config/checkpoint,
   - SAM-3D-Body rerun script on HuMMan sample(s).

## Open Questions

- None for this phase; render-mode and entry-script structure are fixed by decisions above.
