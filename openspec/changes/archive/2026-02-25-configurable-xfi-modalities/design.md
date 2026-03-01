## Context

The current XFi aggregator path in `models/aggregators/xfi_aggregator.py` infers active modalities from whether feature tensors are `None` and relies on hard-coded internal ordering. This creates mismatches between configuration intent and projector branch mapping, which can fail at runtime for valid experiments such as RGB+LiDAR where depth/mmWave are intentionally absent. MMHPE is config-driven, so modality behavior should be explicitly declared and validated in YAML and enforced inside the aggregator.

## Goals / Non-Goals

**Goals:**
- Make XFi aggregator modality usage explicit and configurable from `configs/`.
- Enforce one deterministic modality ordering contract for aggregator branch mapping.
- Fail fast with actionable errors when config-selected modalities and aggregator inputs do not match.
- Preserve compatibility for existing RGB+depth XFi configs by allowing omission of `active_modalities` and deriving it from non-`None` features within the aggregator.
- Document the new config contract in `docs/` with command-line usage examples.

**Non-Goals:**
- No architectural changes to XFi attention/fusion blocks.
- No changes to dataset semantics beyond existing modality tensors.
- No new sensor modality introduction.
- No changes to metric definitions, loss behavior, or logging layout.

## Decisions

1. Add explicit `active_modalities` to XFi aggregator config.
- Rationale: This is the minimal explicit contract for config-driven experiments and avoids hidden coupling to `None` checks.
- Alternatives considered:
- Infer modalities only from non-`None` tensors. Rejected because it is unstable with optional branches and mixed dataset/model settings.
- Infer modalities only from existing backbone blocks. Rejected because runtime input tensors may still be absent/misaligned.

2. Normalize modality ordering to a single canonical list.
- Canonical order: `['rgb', 'depth', 'mmwave', 'lidar']`.
- Rationale: Matches current projector branch layout and prevents index mismatch bugs.
- Alternatives considered:
- Keep current mixed ordering and remap dynamically. Rejected as unnecessarily error-prone.

3. Validate modality consistency inside aggregator forward path.
- Validate that every configured active modality has a non-`None` feature tensor at aggregation time.
- Validate feature tensor shape expectations per modality before branch projection.
- Rationale: fail-fast principle while avoiding changes in `model_api.py`.

4. Keep backward compatibility via default behavior.
- If `active_modalities` is absent in older configs, derive it from non-`None` feature inputs in canonical order and warn once.
- Rationale: avoids breaking existing runs while allowing explicit migration.
- Alternatives considered:
- Make `active_modalities` mandatory immediately. Rejected to avoid broad config churn.

5. Documentation update in `docs/`.
- Add an XFi modality configuration guide with valid combinations and examples for common runs.

## Risks / Trade-offs

- [Risk] Legacy configs may rely on accidental behavior from current ordering.
- Mitigation: canonical ordering + compatibility derivation + explicit validation errors indicating exact mismatch.

- [Risk] Additional validation may surface new aggregation-time failures in previously silent misconfigurations.
- Mitigation: make errors actionable and include expected modalities and received feature presence in message text.

- [Risk] Minor maintenance burden for keeping config docs/examples synchronized.
- Mitigation: include one dedicated docs file and update `configs/baseline/unseen_view_generalization/humman_xfi.yml` in the same change.
