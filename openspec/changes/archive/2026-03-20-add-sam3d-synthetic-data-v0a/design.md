## Context

MMHPE already contains the core ingredients needed for a first synthetic RGB-to-LiDAR data path:
- an established canonical training-side contract for RGB, LiDAR-like point clouds, keypoints, and camera encodings,
- an existing SAM-3D-Body submodule and environment contract,
- visualization tooling and docs for inspecting 3D outputs,
- strong assumptions around single-person samples, fail-fast validation, and config-driven execution.

What is missing is a producer pipeline that starts from a single-image RGB dataset and emits small-scope synthetic artifacts that can later be adapted into MMHPE-compatible training samples. The first milestone is intentionally narrow: `v0-a` should prove that the path works end-to-end on COCO val at `/opt/data/coco` for sample generation and visual inspection, without yet integrating the synthetic data into `main.py` training flows.

This is a cross-cutting change because it touches third-party inference (`third_party/sam-3d-body`), synthetic sensor generation, output contracts, and visualization. It also benefits from explicit decisions before coding because the boundary between "debug artifact" and "future training sample" must be chosen early.

## Goals / Non-Goals

**Goals:**
- Add a dedicated synthetic-data subproject that can run an end-to-end `v0-a` pipeline on single RGB images.
- Use COCO val from `/opt/data/coco` as the initial source dataset.
- Produce the following for each accepted sample:
  - source RGB image/crop/mask metadata,
  - SAM-3D-Body mesh and 3D keypoints,
  - one sampled virtual LiDAR camera/extrinsic,
  - one synthesized LiDAR-style point cloud,
  - visualization outputs for quick human inspection,
  - metadata sufficient to reproduce the generated artifact.
- Keep the generated data aligned with existing MMHPE coordinate/camera conventions where practical, especially pelvis-centered canonical 3D supervision.
- Make quality filtering explicit so failed or low-quality reconstructions are rejected with actionable reasons.

**Non-Goals:**
- No integration into `main.py`, `datasets/data_api.py`, or `models/model_api.py` in `v0-a`.
- No promise of large-scale dataset generation throughput.
- No full physically accurate LiDAR beam simulation in `v0-a`.
- No multi-person scene handling, temporal data generation, or multi-view source dataset support.
- No commitment yet to a final on-disk dataset layout for mixed synthetic + real training runs.

## Decisions

1. Build this as a producer-side subproject inside the repository, not as a new standalone repo.
Why:
- the generated outputs are meant to match MMHPE’s existing training and evaluation contracts,
- the project already contains the relevant camera, coordinate, and visualization conventions,
- keeping it local reduces drift between generator outputs and consumer expectations.
Design:
- place new code under a dedicated area such as `projects/synthetic_data/` with submodules for adapters, SAM-3D-Body inference, virtual sensors, exporters, and visualization.
Alternative:
- create a separate repository now.
Rejected because the lifecycle is still tightly coupled to MMHPE and `v0-a` is exploratory.

2. Keep `v0-a` outside the current training pipeline.
Why:
- the first unknown is synthetic sample quality, not training integration,
- mixing generation and training changes in one milestone makes failures ambiguous.
Design:
- `v0-a` produces saved artifacts and visualization only,
- later milestones can add a dataset adapter/exporter into MMHPE training format after qualitative validation.
Alternative:
- wire synthetic generation directly into `datasets/` now.
Rejected because it introduces churn before the artifact contract is validated.

3. Use a strict staged pipeline: source selection -> full-image mask generation -> SAM-3D-Body -> quality filter -> canonicalization -> virtual LiDAR -> point-cloud synthesis -> export -> visualization.
Why:
- each stage has different failure modes and needs separate debugging outputs,
- this allows rejection before expensive downstream processing.
Alternative:
- a single monolithic script that only emits final outputs.
Rejected because it hides failure modes and makes visual QA difficult.

4. Treat person masking as the explicit preprocessing stage for `v0-a` and do not crop the image before SAM-3D-Body inference.
Why:
- COCO already provides person segmentations for many samples,
- keeping the original image frame preserves simpler traceability between source image, saved mask, and downstream visualization,
- SAM-3D-Body can consume the saved mask as auxiliary input, which is a better `v0-a` fit than introducing a crop-first path immediately.
Design:
- support dataset-provided person selection first,
- save the selected person mask in full-image coordinates,
- pass the full image plus saved mask into SAM-3D-Body,
- permit optional mask-generation fallback only when source mask is missing or unusable.
Alternative:
- crop the selected person before reconstruction.
Rejected for `v0-a` because it adds another coordinate transform layer and is not required for the first end-to-end validation.

5. Canonicalize generated 3D supervision toward MMHPE’s pelvis-centered contract.
Why:
- later training use depends more on coordinate consistency than on matching the original image-space reconstruction frame,
- current MMHPE datasets and metrics already assume canonical/root-centered supervision in multiple places.
Design:
- keep original SAM-3D-Body outputs in metadata,
- derive a pelvis-centered canonical keypoint representation for synthetic supervision artifacts,
- record both canonical and source camera/sensor transforms needed for debugging.
Alternative:
- keep only the raw SAM-3D-Body frame.
Rejected because it complicates future reuse in MMHPE.

6. For `v0-a`, synthesize LiDAR as visible surface point sampling from one virtual sensor, not as a full beam-accurate simulator.
Why:
- the immediate goal is to validate semantic usefulness and coordinate consistency,
- a beam-accurate simulator would add substantial complexity before basic sample quality is known.
Design:
- define one virtual LiDAR extrinsic relative to the reconstructed human,
- determine visible mesh surface regions from that viewpoint,
- sample/subsample a point set and save it as LiDAR-style point cloud output.
Alternative:
- implement full raycasting, beam pattern, dropout, and range noise now.
Deferred to later milestones after `v0-a`.

7. Make visualization a first-class output of the generator.
Why:
- synthetic data quality cannot be trusted from numeric outputs alone,
- the project needs a fast path to reject bad samples before any training experiment.
Design:
- save visual diagnostics per sample:
  - source RGB,
  - saved mask over the source image,
  - SAM-3D-Body mesh overlay,
  - 3D keypoint view,
  - virtual LiDAR pose,
  - synthetic point cloud rendering.
Alternative:
- save only raw tensors/files.
Rejected because it slows iteration and makes QA harder.

8. Start with explicit reproducibility metadata and fail-fast filtering.
Why:
- synthetic generation depends on nontrivial runtime state: checkpoints, crop strategy, virtual sensor parameters, and filtering thresholds,
- without metadata the artifacts are hard to compare or regenerate.
Design:
- record source image id/path, crop/mask provenance, checkpoint root, generation parameters, rejection reason if filtered, and output paths.
Alternative:
- rely on ad hoc folder naming.
Rejected because it is brittle.

## Risks / Trade-offs

- [SAM-3D-Body reconstruction quality varies on in-the-wild RGB] -> Mitigation: add strict rejection criteria and preserve intermediate visualization outputs for manual audit.
- [Synthetic LiDAR point clouds are too unrealistic to help training] -> Mitigation: keep `v0-a` limited to sample validation first; defer training integration until outputs look plausible.
- [COCO val contains multi-person/truncated/occluded scenes that break the pipeline] -> Mitigation: start with one-person filtering and explicit full-image mask quality checks.
- [Canonicalization introduces frame mismatches with raw SAM-3D-Body outputs] -> Mitigation: save both canonical outputs and source-frame metadata for traceability.
- [CUDA/runtime friction from SAM-3D-Body slows iteration] -> Mitigation: reuse the existing preflight/environment contract and keep the first pipeline small-scope.
- [Repository sprawl from synthetic-data code] -> Mitigation: isolate code under one dedicated subproject path and keep training/runtime modules untouched in `v0-a`.

## Migration Plan

1. Add the OpenSpec artifacts and document the `v0-a` scope and sample contract.
2. Create the synthetic-data subproject skeleton and a minimal CLI/config entrypoint.
3. Implement a COCO val input adapter with single-person selection and full-image mask extraction/saving.
4. Reuse the repository SAM-3D-Body environment contract to run inference and save mesh/keypoint outputs.
5. Add virtual LiDAR sampling and visible-surface point-cloud synthesis for one sensor viewpoint.
6. Add visualization and metadata export for one-sample and small-batch inspection.
7. Validate generated sample artifacts before any training integration work.

Rollback is simple because `v0-a` does not alter the current training or dataset runtime path; removing the new subproject leaves existing MMHPE flows untouched.

## Open Questions

- Which exact on-disk output root should become the default for synthetic artifacts: `logs/`, a new `data/synthetic/`, or a configurable path outside the repo?
- For COCO val, should the first filter rely only on annotation-derived single-person masks, or also accept generated mask fallbacks when annotations are incomplete?
- Which minimum quality gates are mandatory for `v0-a`: reprojection error only, or also body-scale/depth/truncation/mask-area heuristics?
