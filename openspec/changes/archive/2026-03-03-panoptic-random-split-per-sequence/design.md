## Context

Panoptic dataset loading is handled in `datasets/panoptic_preprocessed_dataset_v1.py` and instantiated through `datasets/data_api.py` via config-driven parameters from `configs/`.

Current behavior in `_resolve_split_selection()` treats `split_to_use: random_split` as a deterministic random permutation of sequence names, then assigns entire sequences to train or val/test using `ratio` and `random_seed`. This creates two issues for sequence-focused workflows:

- A sequence in `sequence_allowlist` can be absent from val/test entirely.
- The name `random_split` is ambiguous and does not reflect desired frame-level partitioning.

The requested direction is:

- add `temporal_split` for per-sequence temporal split (first 80% train, last 20% val/test);
- make sequence-level partitioning explicit via sequence lists in `cross_subject_split`.

This affects synchronized multimodal Panoptic samples (`rgb`, `depth`, and depth-derived `lidar`) because split is applied on shared synchronized `frame_ids`.

## Goals / Non-Goals

**Goals:**
- Redefine Panoptic `random_split` semantics to deterministic per-sequence temporal split.
- Preserve sequence-level split capability through explicit `sequences` lists in `cross_subject_split` config entries.
- Keep split deterministic and reproducible without dependence on runtime randomness.
- Keep integration incremental: no `main.py`/trainer/model API changes; only dataset split logic and split config contract updates.
- Document split semantics directly in `configs/datasets/panoptic_split_config.yml` comments.

**Non-Goals:**
- No changes to preprocessing output format or synchronization metadata structure.
- No changes to model architecture, losses, metrics, or logging format in `logs/`.
- No new external dependency.
- No changes to non-Panoptic dataset classes.

## Decisions

### 1) `temporal_split` is temporal per-sequence head/tail split

Decision:
- In `PanopticPreprocessedDatasetV1`, `split_to_use == "temporal_split"` will select all candidate sequences (after existing filters), then split each sequence's synchronized `frame_ids` by ordered index:
  - train split (`test_mode=False`): first `floor(ratio * N)` frames
  - val/test split (`test_mode=True`): remaining `N - floor(ratio * N)` frames

Rationale:
- Matches user requirement exactly: first 80% train, last 20% val/test.
- Deterministic and easy to reason about in visual debugging.
- Keeps every sequence available in both partitions when frame count permits.

Alternative considered:
- Randomly shuffle frame IDs within each sequence before split. Rejected because it breaks temporal continuity and complicates sequence-based evaluation/visualization.

### 2) Sequence-level partitioning is explicit list-based `cross_subject_split`

Decision:
- `cross_subject_split.train_dataset.sequences` and `cross_subject_split.val_dataset.sequences` become the source of truth for the existing sequence-level partition behavior.
- `configs/datasets/panoptic_split_config.yml` will include the explicit current train/val sequence lists and comments explaining intent.

Rationale:
- Explicit lists remove ambiguity and prevent hidden split drift.
- Keeps old experiment behavior available without relying on overloaded `random_split` semantics.

Alternative considered:
- Keep using `random_split` name for temporal behavior. Rejected because `temporal_split` is more precise and avoids ambiguity.

### 3) Split application order and modality consistency

Decision:
- Preserve current validation/filter order, then apply temporal split after synchronized `frame_ids` are computed.
- Temporal split is applied once on common frame IDs shared by selected cameras/modalities, so `rgb`, `depth`, and derived `lidar` remain aligned.

Rationale:
- Avoids modality drift and keeps current synchronization guarantees.
- Minimizes code churn: frame IDs are already centralized in `_index_sequences()`.

Alternative considered:
- Per-modality split before intersection. Rejected because it can produce inconsistent frame sets across modalities.

### 4) Failure policy: fail-fast defaults stay strict

Decision:
- Keep fail-fast behavior for invalid ratio bounds and empty split partitions under `strict_validation=True`.
- For very short sequences where one side would be empty, raise explicit error in strict mode; in non-strict mode, skip sequence with warning aggregation (consistent with existing non-strict handling style).

Rationale:
- Aligns with project engineering principle: fail fast on abnormal inputs.

Alternative considered:
- Force at least one frame per partition automatically. Rejected to avoid hidden data-contract mutations.

### 5) Backward-compatibility strategy

Decision:
- Keep current sequence-level train/val partition as explicit `cross_subject_split` sequence lists in `configs/datasets/panoptic_split_config.yml`.
- Migrate Panoptic configs that need frame-level split to `split_to_use: temporal_split`.
- Leave `random_split` documented as legacy/non-preferred for Panoptic configs to prevent semantic ambiguity.

Rationale:
- Prevents silent behavior mismatch from legacy assumptions.

## Risks / Trade-offs

- [Behavior drift across configs due to mixed split names] -> Mitigation: make `temporal_split` the documented Panoptic default for frame-level split and keep explicit `cross_subject_split.sequences` for sequence-level split.
- [Short sequences may fail partition constraints] -> Mitigation: explicit validation error with actionable message including sequence name, `N`, and computed cutoff.
- [Potential confusion between val/test naming and `test_mode`] -> Mitigation: keep current contract (`val_dataset` used when `test_mode=True`) and document that val/test share the same held-out temporal tail policy.
- [Different datasets roots can have different available sequence sets] -> Mitigation: explicit sequence lists in config for sequence-level split; temporal split requires no global sequence permutation.

## Migration Plan

1. Update `configs/datasets/panoptic_split_config.yml`:
- Add comments defining:
  - `temporal_split`: per-sequence temporal head/tail split.
  - `cross_subject_split`: explicit list-based sequence split for legacy sequence partition behavior.
- Populate `cross_subject_split.train_dataset.sequences` and `cross_subject_split.val_dataset.sequences` with the currently used train/val sequence partition.

2. Update dataset logic in `datasets/panoptic_preprocessed_dataset_v1.py`:
- Add or reuse a dedicated `temporal_split` branch for frame-level splitting.
- Carry temporal-split config through split resolution and apply on synchronized `frame_ids` per sequence.
- Add explicit ratio/empty-partition validation errors.

3. Validate with Panoptic visualization/training configs:
- Confirm `sequence_allowlist` intersections are non-empty for expected splits.
- Confirm temporal split keeps both train and val/test samples per sequence when frame count allows.

4. Rollback strategy:
- Revert dataset split logic and split config comments/lists to previous commit if downstream regressions are observed.

## Open Questions

- None for this change scope. `temporal_split` is adopted as the explicit frame-level split name, and no additional minimum val/test frame threshold is required beyond existing validations.
