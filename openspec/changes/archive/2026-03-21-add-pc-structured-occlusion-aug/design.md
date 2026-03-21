## Context

The current synthetic and real-data training pipelines already apply point-cloud normalization and padding transforms at runtime. That makes runtime augmentation the natural place to simulate external-object occlusions or sensor dropouts that are not part of the deterministic sensor-formation process. Baking such occlusion into saved datasets would create many dataset variants and make ablations harder.

This change needs to stay compatible with the existing config-driven pipeline model in `datasets/data_api.py` and the current RGB+LiDAR training configs. It also needs to preserve the GT contract: the augmentation changes `input_lidar`, while GT keypoints remain in the same underlying coordinate frame expected by downstream transforms.

## Goals / Non-Goals

**Goals:**
- Add a reusable runtime transform for structured LiDAR occlusion/dropout.
- Support contiguous range-image blob occlusion patterns that look more realistic than per-point Bernoulli dropout.
- Make the transform configurable from YAML, including probability and severity parameters.
- Keep the augmentation compatible with the current PC-centering and padding workflow.

**Non-Goals:**
- Changing saved synthetic dataset artifacts.
- Simulating self-occlusion from the body mesh; that belongs to the `v1` LiDAR generation change.
- Building a full physics-based simulator for external scene geometry.
- Modifying model heads or loss functions directly as part of this change.

## Decisions

### 1. Implement augmentation as a runtime dataset transform
Add a point-cloud transform under `datasets/transforms/` that mutates `input_lidar` during training only.

Why:
- It avoids dataset regeneration.
- It supports stochastic augmentation across epochs.
- It keeps ablations simple because configs can toggle it on or off.

Alternatives considered:
- Baking external occlusion into saved synthetic roots: too rigid and storage-heavy.

### 2. Use range-image blob dropout as the structured occlusion mode
The transform should project the LiDAR points into a range-image-style representation, erase one or more contiguous blobs in that 2D view, and remove the corresponding 3D points from `input_lidar`.

Why:
- Real occlusions remove regions, not uniformly random isolated points.
- Range-image blobs produce more natural-looking missing scan regions than independent point dropout.
- It stays tied to LiDAR observation structure while remaining generic and configurable.

Alternatives considered:
- Angular-sector dropout: simpler, but more rigid and less representative of irregular missing regions.
- Pointwise Bernoulli dropout only: too artificial for the stated goal.

### 3. Run the augmentation before PC centering
The transform should run on the raw `input_lidar` points before `PCCenterWithKeypoints`.

Why:
- The occlusion is meant to mimic missing observations in the sensor frame.
- Centering should operate on the already-occluded point cloud so downstream GT centering remains consistent with the actual input.
- This keeps the augmentation generic across synthetic and real LiDAR samples.

Alternatives considered:
- Apply after PC centering: mixes the sensor-formation augmentation with a derived normalized frame and makes the augmentation less physically interpretable.

### 4. Keep augmentation parameters explicit in configs
Expose parameters such as:
- probability of applying augmentation
- blob count
- blob size or radius in range-image coordinates
- optional shape/jitter parameters
- training-only enablement

Why:
- This project is heavily config-driven.
- One generic transform with configurable parameters is enough; separate preset-specific transforms are unnecessary.

Alternatives considered:
- Hard-coded augmentation policy: too inflexible for transfer experiments.

## Risks / Trade-offs

- [Augmentation becomes too destructive and harms training] → Mitigation: require explicit probability/severity parameters and document conservative defaults.
- [Transform order conflicts with PC-centering/padding] → Mitigation: fix the expected position before `PCCenterWithKeypoints` and validate with existing pipelines.
- [Synthetic and real pipelines diverge too much] → Mitigation: implement the augmentation as a generic point-cloud transform that can be reused in both.
- [Config complexity increases] → Mitigation: provide example config snippets and keep the first version to one generic range-image blob mode with parameter tuning.

## Migration Plan

1. Implement the range-image blob occlusion transform under `datasets/transforms/`.
2. Add config snippets or example experiment configs that place the transform before `PCCenterWithKeypoints`.
3. Validate shape and frame compatibility on synthetic and at least one real dataset pipeline.
4. Document recommended usage for synthetic pretraining and finetuning.

Rollback:
- Remove the transform from training configs; no dataset rollback is needed because artifacts remain unchanged.

## Resolved Parameters

- The first version supports both axis-aligned rectangles and circular blobs, with a `mixed` mode exposed in YAML so experiments can choose one shape family or alternate between both.
