## Context

H36M multiview training can be configured to use only keypoint inputs, but the current
H36MMultiViewDataset always loads RGB frames into memory. This adds I/O and RAM overhead
without affecting model outputs for keypoint-only experiments. The change is scoped to
H36MMultiViewDataset only and should not impact other datasets.

## Goals / Non-Goals

**Goals:**
- Add a dataset-level toggle to skip RGB frame loading in H36MMultiViewDataset.
- Preserve all non-image data (keypoints, camera parameters, sample IDs) so downstream
  modules can run unchanged.
- Keep GPU usage unaffected; the change is CPU/I/O only.

**Non-Goals:**
- Do not change other datasets or global pipeline/transforms behavior.
- Do not add new dependencies or change preprocessing outputs.
- Do not implement broader image caching or compression schemes.

## Decisions

- **Add a `load_rgb` flag to H36MMultiViewDataset (default: true).**
  This keeps existing configs working and allows keypoints-only configs to disable
  image loading. Alternative considered: adding a pipeline transform to drop images.
  Rejected because image decoding would still happen and wastes I/O.

- **When `load_rgb` is false, skip cv2 reads and set `input_rgb` to None (or omit it).**
  This avoids disk I/O and memory for images while keeping the sample structure stable.
  Alternative considered: return a zero tensor placeholder. Rejected because it still
  allocates large arrays and wastes memory.

- **Leave camera parameters intact regardless of `load_rgb`.**
  Needed for GT 2D projection or other geometry-dependent transforms.

## Risks / Trade-offs

- **Risk:** Some transforms may assume `input_rgb` is always present.
  → **Mitigation:** Update H36M configs/pipelines to avoid RGB-dependent transforms
  when `load_rgb=false`, and document the expected behavior in configs.

- **Risk:** Downstream model code expects `input_rgb` key to exist.
  → **Mitigation:** Keep the key but set it to None, or gate in data collate
  (decide during implementation based on pipeline usage).
