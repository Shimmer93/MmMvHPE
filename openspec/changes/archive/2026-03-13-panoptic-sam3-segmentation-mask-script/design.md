## Context

Panoptic sequence data is stored per sequence with camera-specific RGB images already organized under the raw dataset directory. The new requirement is not a training-time model change; it is a dataset-side asset generation step that should populate reusable segmentation masks next to the raw sequence content.

The repository already has a working SAM3 integration path through the vendored SAM-3D-Body code:
- `third_party/sam-3d-body/tools/build_sam.py` exposes `HumanSegmentor(name="sam3", ...)`
- its `run_sam3(...)` path applies the text prompt `person`
- it returns one mask per confident detected person instance

This means the new script can reuse the existing SAM3 integration instead of introducing another segmentation stack or another prompt formulation. The main work is dataset traversal, output layout, binary mask materialization, and incremental execution controls.

Constraints:
- The script should fit the existing repository style: `uv run python ...`, code under `tools/`, docs under `docs/`.
- Output goes into the Panoptic dataset tree, not `logs/`.
- The Panoptic dataset can be partially downloaded, so the script must support selected sequences/cameras and fail clearly on malformed paths.
- The user wants masks unioned across all people in a frame rather than one mask per person.

## Goals / Non-Goals

**Goals:**
- Provide a standalone script that traverses Panoptic RGB images and writes one binary segmentation mask per RGB image.
- Reuse the same SAM3 prompt behavior as SAM-3D-Body rather than approximating it with another detector/segmentor path.
- Support incremental runs over selected sequences and selected Kinect cameras.
- Materialize outputs deterministically under `<sequence>/sam_segmentation_mask/<camera>/`.
- Preserve a one-to-one mapping between RGB images and mask files, with filenames derived from the RGB inputs.
- Fail fast on missing sequences, missing RGB camera folders, unreadable images, or missing SAM3 checkpoints.

**Non-Goals:**
- No integration into `main.py` or the training loop.
- No attempt to denoise, postprocess, or temporally smooth masks beyond binary union of all person masks returned by SAM3.
- No support for non-Panoptic datasets in this change.
- No requirement to use dataset classes or YAML configs; this is a direct filesystem tool, not a training/evaluation pipeline component.
- No instance-segmentation output, confidence maps, polygons, or per-person mask files.

## Decisions

### 1. Implement as a direct filesystem tool under `tools/`

Decision:
- Add a dedicated script under `tools/`, tentatively `tools/generate_panoptic_sam3_segmentation_masks.py`.

Rationale:
- The output target is the raw dataset tree, not a model artifact directory.
- The task is sequence traversal plus file generation, which is better served by a direct CLI than by dataset classes intended for training/evaluation samples.
- This keeps the script usable while downloads or preprocessing are still incomplete.

Alternatives considered:
- Build it as a dataset-config-driven runner similar to visualization scripts: rejected because configs add little value for raw per-image export and would make partial-data handling more fragile.
- Integrate into preprocessing script directly: rejected because mask generation is a separate long-running dependency-heavy step that should remain independently rerunnable.

### 2. Reuse `HumanSegmentor(name="sam3")` from SAM-3D-Body

Decision:
- The script should load SAM3 through `third_party/sam-3d-body/tools/build_sam.py` and use `HumanSegmentor(name="sam3", ...)`.

Rationale:
- This is the existing repository path that already matches the user's requirement: prompt with `person`.
- Reusing it avoids divergence between “SAM3DBody-style masks” and the new Panoptic export script.
- It also keeps checkpoint loading behavior aligned with the rest of the repo.

Alternatives considered:
- Call SAM3 directly from a new integration layer: rejected because it duplicates the prompting and checkpoint-loading logic.
- Use detector boxes plus SAM2/SAM3 refinement: rejected because the user explicitly asked for the SAM3DBody-style path, and `run_sam3` already produces masks from the text prompt.

### 3. Union all confident person masks into one binary image

Decision:
- For each RGB frame, the script should threshold exactly as the reused SAM3 path does, union all returned person masks, and write one binary mask image.

Rationale:
- The user wants a single foreground mask even when multiple people are present.
- Union preserves all human regions and keeps the downstream format simple.

Alternatives considered:
- Keep only the largest person: rejected because it would silently drop valid people.
- Save one mask per person: rejected because it does not match the requested directory structure or downstream expectation.

### 4. Output layout is sequence-local and camera-specific

Decision:
- For each sequence root, write masks to:
  - `<sequence>/sam_segmentation_mask/<kinect_camera>/<mask_file>`
- The script should mirror the RGB camera coverage it processes and keep output separated by camera.

Rationale:
- This matches the user’s requested storage layout.
- It makes the generated masks easy to inspect and reuse from sequence-local preprocessing code.

Alternatives considered:
- Central output root under `logs/`: rejected because these are dataset assets, not experiment artifacts.
- One flat directory per sequence without camera subfolders: rejected because Panoptic is multiview and file names alone are not enough structure.

### 5. Filename mapping should preserve RGB identity, but the on-disk image format must remain lossless

Decision:
- The script should preserve the RGB basename exactly and keep a deterministic mask filename mapping.
- If the source RGB extension is already lossless (for example `.png`), use the same full filename.
- If the source RGB extension is lossy (for example `.jpg` / `.jpeg`), write a lossless mask file with the same stem and `.png` extension, and document this explicitly.

Rationale:
- Binary masks should not be written as JPEG because compression artifacts would corrupt the mask.
- The user’s intent is clear: masks should correspond one-to-one with RGB images. Preserving basename identity is the important part.

Alternatives considered:
- Force exact filename including `.jpg`: rejected because lossy storage is not acceptable for binary segmentation masks.
- Store NumPy arrays only: rejected because image files are easier to inspect and align with the user’s requested layout.

### 6. Incremental execution must support skip-existing mode

Decision:
- The CLI should support targeting specific sequences and cameras and should skip already-written mask files by default, with an explicit overwrite switch.

Rationale:
- Panoptic data is large and partially downloaded; rerunning from scratch is wasteful.
- Skip-existing is the safest default for long-running export jobs.

Alternatives considered:
- Always overwrite: rejected because it increases runtime and makes interrupted jobs harder to resume.
- Track progress in an external database: rejected as unnecessary operational complexity.

### 7. Fail fast on path and dependency assumptions

Decision:
- The script should raise explicit errors for:
  - missing sequence directories
  - missing RGB camera directories
  - missing SAM3 checkpoints
  - unreadable/corrupt RGB images
  - empty RGB image matches after filtering

Rationale:
- This repository’s engineering rule is to fail fast on abnormal inputs.
- Silent skips would make it too easy to think masks were generated when part of the dataset was never processed.

Alternatives considered:
- Continue-on-error by default: rejected because missing data should be visible immediately.

## Risks / Trade-offs

- [SAM3 inference is slow on large Panoptic sequences] → Provide sequence/camera filters, skip-existing behavior, and progress logging.
- [SAM3 checkpoint or package installation may be incomplete on some machines] → Validate imports and checkpoint paths at startup before scanning images.
- [Binary mask filename compatibility is ambiguous when RGB inputs are JPEG] → Use lossless `.png` for lossy RGB inputs and document the exact mapping.
- [Unioned masks may include bystanders or multiple people when only the main subject is desired] → Keep the implementation aligned with the user’s explicit requirement to union all detected people.
- [Dataset directory structure may vary across partial downloads] → Require explicit sequence roots and camera folders to exist before processing and stop on mismatches.

## Migration Plan

1. Add the new script under `tools/`.
2. Add documentation under `docs/` with environment requirements, checkpoint expectations, directory conventions, and command examples.
3. Validate on one known Panoptic sequence/camera before broader use.
4. Run incrementally over selected sequences/cameras to populate `sam_segmentation_mask/`.

Rollback:
- Since outputs are isolated under `sam_segmentation_mask/` within each sequence, rollback is operationally simple: remove that generated directory for the affected sequence/camera.

## Open Questions

- Should the script expose an optional `--continue-on-error` mode for long batch runs, or should it stay strict-only in the first version?
- Which exact Panoptic RGB directory pattern should be treated as the canonical input layout for this project if multiple raw layouts are present?
- Does any downstream preprocessing code require exact filename extension parity with RGB images, or is basename parity plus lossless `.png` acceptable?
