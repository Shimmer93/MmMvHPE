## Context

`PCMPJPE` and `SMPL_PCMPJPE` currently align both pelvis translation and orientation before error computation. The requested change is a behavioral redefinition: keep translation centering, but retain rotation differences in the error term.

## Goals / Non-Goals

**Goals:**
- Make `PCMPJPE` and `SMPL_PCMPJPE` translation-only centered metrics.
- Keep existing metric names and config wiring intact.
- Update tests and docs to match new behavior.

**Non-Goals:**
- Renaming metrics or adding new metric variants.
- Changing dataset coordinate transforms.
- Touching training losses or model outputs.

## Decisions

1. Remove root-orientation alignment step from `pcmpjpe_func`.
- Rationale: this directly enforces translation-only semantics.
- Alternative considered: add a flag to toggle rotation alignment. Rejected to keep behavior strict and simple per request.

2. Keep metric interfaces callable from existing configs.
- Rationale: avoids config churn and keeps backward compatibility at the integration level.
- Tradeoff: semantic break under same metric name is intentional and documented as breaking.

3. Reframe tests around translation-only invariance.
- Rationale: tests should assert that translation-only offsets collapse, while rotation-only offsets do not.

## Risks / Trade-offs

- [Risk] Historical comparisons become inconsistent across runs before/after change -> Mitigation: document breaking semantic change explicitly.
- [Risk] Some existing tuning that assumed orientation cancellation will shift -> Mitigation: no API/config change required; only interpretation changes.
