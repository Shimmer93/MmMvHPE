## Context

We will use the existing Mamba4D point‑cloud encoder (`models/pc_encoders/mamba4d.py`) as the
backbone for a point‑cloud‑only 3D pose estimator on HuMMan. The estimator must output a
sequence of 3D keypoints to feed FusionFormer. The current heads assume single‑pose output,
so we need a sequence head and sequence loss while keeping the codebase stable and modular.

## Goals / Non-Goals

**Goals:**
- Add a sequence regression head that outputs `B, T, J, 3` with uniform per‑frame MSE loss.
- Keep Mamba4D backbone untouched; regress directly from its per‑frame features.
- Provide config wiring for HuMMan point‑cloud‑only training.

**Non-Goals:**
- No end‑to‑end FusionFormer training in this change (focus on the PC pose estimator output).
- No new dependencies or dataset preprocessing changes.

## Decisions

- **Sequence head + uniform MSE loss.**
  The head will regress all frames’ keypoints; loss is averaged across T and J.

- **No temporal conv by default.**
  We will directly regress from Mamba4D per‑frame features to minimize changes and risk.

- **Config‑driven integration.**
  New config selects Mamba4D backbone + sequence head for HuMMan.

## Risks / Trade-offs

- **Risk:** Shape mismatches across time dimensions.
  → **Mitigation:** Add explicit shape assertions in the head.

- **Risk:** If Mamba4D lacks temporal context, results may be noisier.
  → **Mitigation:** Add an optional temporal conv later if needed.
