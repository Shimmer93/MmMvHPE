## 1. Project Skeleton and CLI

- [x] 1.1 Create a dedicated synthetic-data subproject area (for example `projects/synthetic_data/`) with modules for source adapters, SAM-3D-Body inference, virtual sensors, export, and visualization.
- [x] 1.2 Add a runnable `v0-a` entry script/CLI that targets COCO val from `/opt/data/coco` and writes outputs to a configurable artifact root.
- [x] 1.3 Add fail-fast config/argument validation for required source paths, checkpoint paths, and output paths.

## 2. COCO Input and Preprocessing

- [x] 2.1 Implement a COCO val input adapter that can select one target human from a source image and expose image id/path plus annotation-derived mask metadata.
- [x] 2.2 Implement full-image mask extraction/saving for the selected person with explicit saved intermediate outputs and mask provenance metadata.
- [x] 2.3 Add rejection rules for unusable source samples (for example no valid target person, invalid mask, or missing required annotation inputs).

## 3. SAM-3D-Body Reconstruction and Quality Filtering

- [x] 3.1 Reuse the repository SAM-3D-Body environment/checkpoint contract to run reconstruction on the full source image with the saved person mask as auxiliary input.
- [x] 3.2 Save reconstructed mesh, 3D keypoints, and reconstruction metadata needed for later synthetic processing.
- [x] 3.3 Implement quality gates and explicit rejection reasons for failed or implausible reconstructions.
- [x] 3.4 Canonicalize accepted 3D keypoints into a pelvis-centered output contract while preserving source-frame traceability metadata.

## 4. Virtual LiDAR and Synthetic Point Cloud

- [x] 4.1 Implement one virtual LiDAR pose sampler for `v0-a` with reproducible pose metadata.
- [x] 4.2 Implement visible-surface mesh sampling from the virtual LiDAR viewpoint to produce one LiDAR-style point cloud per accepted sample.
- [x] 4.3 Save synthetic point cloud artifacts together with simulation mode and sampling parameters.

## 5. Visualization and Artifact Packaging

- [x] 5.1 Implement visualization outputs covering source RGB, full-image saved mask, reconstruction overlay, 3D keypoints, virtual LiDAR context, and synthetic point cloud.
- [x] 5.2 Define and save per-sample metadata/artifact manifests that identify acceptance status, rejection reason when applicable, and output file paths.
- [x] 5.3 Add a small-batch mode for generating and reviewing multiple `v0-a` samples without changing scope to large-scale dataset generation.

## 6. Validation and Documentation

- [x] 6.1 Add smoke validation commands using `uv run` for one-sample generation and small-batch inspection on COCO val.
- [x] 6.2 Verify that accepted samples contain the required intermediate artifacts, canonical keypoints, virtual LiDAR metadata, and synthetic point cloud outputs.
- [x] 6.3 Add/update docs in `docs/` describing the `v0-a` pipeline stages, assumptions, output structure, and runnable examples.
- [x] 6.4 Record current limitations and follow-up milestones (`v0-b`, `v0-c`, `v1`) in documentation so future work stays aligned with the saved plan.
