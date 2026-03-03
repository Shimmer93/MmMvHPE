## Context

MMHPE currently has HuMMan dataset classes (including `humman_dataset_v3`) that are already integrated into config-driven training workflows. We now have preprocessed Panoptic Kinoptic single-actor data under `/opt/data/panoptic_kinoptic_single_actor_cropped`, but there is no dataset class that exposes a training-facing interface compatible with `humman_dataset_v3`.

Without interface alignment, training code and configs must branch per dataset, increasing maintenance cost and raising regression risk. The design should therefore integrate Panoptic incrementally in `datasets/` and config wiring paths (`datasets/data_api.py`, config YAMLs), with minimal churn to model/loss/metric code.

Constraints:
- Fail fast on malformed/incomplete sequence data.
- Preserve sequence-local semantics.
- Keep behavior strict for single-actor preprocessed Panoptic format.
- Default data root should be `/opt/data/panoptic_kinoptic_single_actor_cropped`.

## Goals / Non-Goals

**Goals:**
- Add a Panoptic preprocessed dataset class with `humman_dataset_v3`-compatible sample interface for train/eval/predict.
- Support config-driven selection of split/sequences and robust filtering of incomplete sequences.
- Reuse existing pipeline entrypoints (`main.py` via `datasets/data_api.py`) without model API changes.
- Keep modality support aligned to available preprocessed artifacts (RGB, depth, GT keypoints, cameras metadata).

**Non-Goals:**
- Multi-actor Panoptic support.
- Raw Panoptic/Kinoptic decoding in the dataset class (it must consume preprocessed outputs only).
- Broad refactor of existing HuMMan dataset classes.
- New external runtime dependencies.

## Decisions

1. **Create a dedicated dataset class instead of overloading HuMMan classes**
- Decision: add a new class (for example `PanopticPreprocessedDatasetV1`) in `datasets/`.
- Rationale: Panoptic preprocessed structure is sequence-based and differs from HuMMan naming/metadata conventions. A dedicated class avoids brittle conditional logic in `humman_dataset_v3`.
- Alternative considered: add Panoptic branches inside `humman_dataset_v3`.
  - Rejected due to increased complexity and higher regression risk for existing HuMMan runs.

2. **Match `humman_dataset_v3` output contract at sample level**
- Decision: normalize Panoptic-loaded sample dict keys/tensor shapes to mirror what training code expects from `humman_dataset_v3`.
- Rationale: keeps downstream model/loss/metric code unchanged, minimizes integration risk.
- Alternative considered: introduce a separate Panoptic-only sample contract and modify downstream pipeline.
  - Rejected as unnecessary cross-cutting churn.

3. **Integrate through existing dataset registry/api path**
- Decision: wire new dataset via `datasets/data_api.py` and config dataset type entries; no changes to `main.py` logic.
- Rationale: existing config-driven architecture already handles dataset instantiation through data API.
- Alternative considered: custom Panoptic training script.
  - Rejected because it fragments the standard workflow.

4. **Strict sequence validation and skip policy controlled by config/args**
- Decision: validate per-sequence required files (`rgb/`, `depth/`, `gt3d/`, `meta/sync_map.json`, `meta/cameras_kinect_cropped.json`) during index build; fail with explicit errors unless configured to skip invalid sequences.
- Rationale: supports fail-fast principle while still allowing practical use on partially prepared data.
- Alternative considered: broad fallback on missing artifacts.
  - Rejected by engineering principle (avoid broad fallback logic for invalid input).

5. **Camera/modality compatibility adapter at dataset boundary**
- Decision: map Panoptic camera keys and metadata to the same camera key style consumed in existing transforms/model paths.
- Rationale: isolates naming differences in dataset layer; keeps transform/model code stable.
- Alternative considered: update transforms/models to accept Panoptic-native keys.
  - Rejected as avoidable blast radius.

6. **Split configuration follows existing dataset split-config pattern**
- Decision: support split configuration via a config file path in the same style as `configs/datasets/humman_split_config.yml`.
- Rationale: keeps dataset split behavior aligned with current MMHPE config workflow.

7. **Deterministic sequence-level ratio split when lists are absent**
- Decision: if explicit split sequence lists are not provided, derive train/val/test with deterministic sequence-level ratio split (fixed seed + stable sequence ordering).
- Rationale: supports reproducible experiments without requiring manually curated lists for every run.

8. **Keep full `humman_dataset_v3`-compatible field set**
- Decision: keep all interface fields expected by downstream training paths for compatibility, as long as they do not introduce major implementation risk.
- Rationale: reduces integration risk and avoids subtle breakage in existing model/loss/metric consumers.

## Risks / Trade-offs

- **[Interface drift vs `humman_dataset_v3`]** → Mitigation: explicitly compare key set and tensor shape expectations; add a lightweight compatibility assertion test/sample dump.
- **[Incomplete preprocessed sequences in default root]** → Mitigation: strict validation with clear per-sequence diagnostics and optional skip-invalid mode.
- **[Performance overhead from per-sequence metadata parsing]** → Mitigation: precompute sequence/frame index once at init and reuse cached in-memory index.
- **[Backward compatibility with existing configs]** → Mitigation: additive dataset type/config only; no default behavior change to existing HuMMan configs.
- **[CUDA-sensitive environment concerns]** → Mitigation: dataset implementation stays CPU-side and dependency-neutral; no new CUDA dependency introduced.

## Migration Plan

- Add new dataset class and register it in dataset API.
- Add Panoptic dataset config examples under `configs/` referencing default root `/opt/data/panoptic_kinoptic_single_actor_cropped`.
- Add docs for usage and expected preprocessed directory contract.
- Validate by running a short config-driven train/eval dry run.
- Rollback strategy: remove new dataset config usage and dataset registration; existing workflows remain unchanged.

## Open Questions

- None at this stage.
