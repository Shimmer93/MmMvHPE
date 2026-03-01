## Context

FusionFormer expects a pose feature extractor that maps per‑joint embeddings
\(F_{embed} \in \mathbb{R}^{T \times V \times J \times C_J}\) to per‑pose features
\(F^{(0)}_{pose} \in \mathbb{R}^{T \times V \times C_P}\). Our current extractor
is a simple per‑joint projection with mean pooling, which does not model joint
relationships as described in the paper. We will align the extractor to the
EFC baseline (3‑layer FCN) and wire it into the aggregator.

## Goals / Non-Goals

**Goals:**
- Implement the EFC baseline as a 3‑layer FCN operating over joint embeddings.
- Output per‑pose features \(T \times V \times C_P\) as the paper specifies.
- Keep GPU compute localized to the extractor/aggregator modules.

**Non-Goals:**
- Implement PoseFormer or ET baseline in this change.
- Modify dataset preprocessing or pose projection.

## Decisions

- **EFC implemented as MLP over flattened joint features.**
  We will flatten the joint dimension per (T,V) token and apply a 3‑layer FCN
  to obtain pose features. This matches the paper’s definition while keeping
  implementation simple and GPU‑friendly.

- **Expose extractor selection via config.**
  A new extractor module will be selectable in config so we can compare EFC
  against existing extractors without changing other components.

## Risks / Trade-offs

- **Risk:** FCN over flattened joints increases parameter count.
  → **Mitigation:** Keep hidden size modest and configurable.

- **Risk:** Shape mismatches with existing aggregator assumptions.
  → **Mitigation:** Add explicit shape comments and assertions during integration.
