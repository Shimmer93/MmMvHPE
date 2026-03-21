## Context

The existing synthetic pipeline already saves the reconstructed mesh, virtual LiDAR pose, and point-cloud artifacts per sample. That makes self-occlusion a good second-stage upgrade: the expensive upstream stages have already been paid for, and the point-cloud generator can be rerun over the saved synthetic root. The current `v0-a` sampler is viewpoint-aware only through face normals, so it does not reject points hidden by the body itself.

The design needs to preserve three constraints:
- keep the current synthetic sample layout stable enough that existing training/export code does not break;
- avoid rerunning SAM-3D-Body or MHR-to-SMPL conversion;
- make the LiDAR generation mode explicit so `v0-a` and `v1` outputs can coexist or be compared.

## Goals / Non-Goals

**Goals:**
- Regenerate synthetic LiDAR point clouds from saved mesh and sensor artifacts with self-occlusion-aware visibility.
- Use a depth-buffer visibility pass rather than a full beam-accurate LiDAR simulator.
- Support per-sample and dataset-scale regeneration over existing synthetic roots.
- Version saved point-cloud metadata so downstream exporters/loaders can tell which simulation mode produced the data.
- Provide QC outputs that let a user compare prior visible-surface sampling against the new self-occlusion-aware result.

**Non-Goals:**
- Simulating external-object occlusion, structured dropout, or range-image augmentation during this change.
- Recomputing SAM-3D-Body, MHR-to-SMPL, or target-format exports from scratch.
- Implementing a physically accurate spinning-LiDAR beam model.
- Changing the current training dataset contract beyond consuming the regenerated `input_lidar`.

## Decisions

### 1. Use a depth-buffer visibility pass in the virtual sensor frame
Use a depth-image style visibility test instead of normal-only filtering. The mesh will be transformed into the virtual LiDAR sensor frame, rasterized/projected into a depth buffer, and sampled points will only survive if their projected depth matches the front-most surface within a configurable tolerance.

Why:
- It is much simpler than full raycasting or beam simulation.
- It directly addresses self-occlusion, which is the current realism gap.
- It can reuse the already-saved virtual LiDAR pose and mesh.

Alternatives considered:
- Raycasting first-hit visibility: more physically grounded but more engineering work and slower to tune.
- Keep normal-only filtering: too weak, already known to retain hidden surfaces.

### 2. Treat this as a regeneration pass over existing synthetic roots
Implement a script that walks existing sample directories, loads mesh plus sensor metadata, regenerates the LiDAR artifact, and writes the `v1` outputs alongside the existing `v0-a` LiDAR artifacts. The selected LiDAR version will later be chosen by dataset/config parameters rather than by overwriting the old artifact.

Why:
- The synthetic root already contains the necessary upstream artifacts.
- It avoids recomputing the expensive RGB-to-mesh pipeline.
- It keeps the migration path practical for large existing roots like COCO train.

Alternatives considered:
- Full dataset regeneration: too slow and unnecessary.
- Runtime self-occlusion at training time: wrong layer, because self-occlusion is part of the saved sensor formation process.

### 3. Save explicit simulation metadata and optional side-by-side QC outputs
The regenerated sample metadata should record at least:
- simulation version or mode
- depth-buffer resolution
- visibility tolerance
- sampling count before/after visibility filtering
- regeneration timestamp

QC tooling should optionally render old vs new point clouds for selected samples.

Why:
- The current workflow already depends heavily on saved manifests.
- Comparing `v0-a` to `v1` is important before rolling the dataset update into training.

Alternatives considered:
- Implicit overwrite with no version tag: too hard to audit and compare.

### 4. Define depth-buffer resolution as the rendered depth-map size and benchmark it before fixing the default
Here, depth-buffer resolution means the width and height of the rendered depth map used for visibility testing in the virtual sensor frame. The implementation should not hard-code the default resolution before benchmarking at least `512x512`, `720x720`, and `1024x1024` on a validation subset and estimating the full-dataset runtime/quality trade-off.

Why:
- Resolution directly affects both runtime and self-occlusion quality.
- The correct default is an empirical decision, not a purely architectural one.
- The benchmark can be done on a small subset before committing to a full-dataset regeneration.

Alternatives considered:
- Pick `512x512` immediately for speed: too likely to leave visible discretization artifacts.
- Pick `1024x1024` immediately for quality: may be unnecessarily slow at full-dataset scale.

## Risks / Trade-offs

- [Performance on full synthetic roots] → Mitigation: support per-sample and resumable dataset-scale regeneration, and keep the algorithm limited to self-occlusion rather than full beam simulation.
- [Depth-buffer discretization artifacts] → Mitigation: record resolution/tolerance in metadata and validate on QC samples before bulk regeneration.
- [Breaking assumptions in downstream scripts] → Mitigation: preserve the existing `v0-a` artifact, add explicit versioned LiDAR metadata, and update the dataset/config contract so the desired LiDAR version is selected deliberately.
- [Old and new point clouds becoming hard to distinguish] → Mitigation: require explicit simulation mode/version fields in manifests and docs.

## Migration Plan

1. Add the new self-occlusion-aware point-cloud generator and metadata schema.
2. Add a per-sample CLI and a dataset-scale resumable regeneration CLI for existing synthetic roots.
3. Benchmark `512x512`, `720x720`, and `1024x1024` depth-buffer resolutions on a validation subset with optional QC rendering enabled.
4. Choose the default depth-buffer resolution from that benchmark.
5. Regenerate the intended synthetic roots with QC rendering disabled for bulk processing.
6. Point training/export workflows at the updated synthetic roots by selecting the desired LiDAR version from config.

Rollback:
- Keep the old `v0-a` artifacts untouched and continue selecting that LiDAR version from config if the `v1` QC or benchmark result looks wrong.

## Open Questions

- What benchmark result across `512x512`, `720x720`, and `1024x1024` depth-buffer resolutions gives the best runtime versus point-cloud quality trade-off for full-dataset regeneration?
