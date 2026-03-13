## 1. Script Setup

- [x] 1.1 Create the CLI entry script at `tools/generate_panoptic_sam3_segmentation_masks.py` with arguments for `--data-root`, `--sequences`, `--cameras`, `--segmentor-path`, `--overwrite`, and `--continue-on-error`.
- [x] 1.2 Add startup validation in `tools/generate_panoptic_sam3_segmentation_masks.py` for dataset root existence, SAM3 import availability, and checkpoint path availability before image traversal begins.
- [x] 1.3 Add reusable helpers in `tools/generate_panoptic_sam3_segmentation_masks.py` for canonical Panoptic RGB path resolution under `<sequence>/rgb/<camera>/` and output path resolution under `<sequence>/sam_segmentation_mask/<camera>/`.

## 2. RGB Traversal And Mask Generation

- [x] 2.1 Implement sequence and camera filtering in `tools/generate_panoptic_sam3_segmentation_masks.py` so the script can process selected subsets of the Panoptic dataset incrementally.
- [x] 2.2 Reuse `third_party/sam-3d-body/tools/build_sam.py` in `tools/generate_panoptic_sam3_segmentation_masks.py` to construct `HumanSegmentor(name="sam3", ...)` and generate masks with the `person` prompt path.
- [x] 2.3 Implement per-image SAM3 inference in `tools/generate_panoptic_sam3_segmentation_masks.py` that unions all returned confident person masks into one binary foreground mask.
- [x] 2.4 Implement lossless output writing in `tools/generate_panoptic_sam3_segmentation_masks.py` so PNG inputs keep the same filename and lossy RGB inputs write a `.png` mask with the same basename stem.

## 3. Incremental Execution And Failure Handling

- [x] 3.1 Implement skip-existing behavior by default in `tools/generate_panoptic_sam3_segmentation_masks.py` and add explicit overwrite support.
- [x] 3.2 Implement strict-mode failure behavior for missing sequence paths, missing camera paths, unreadable RGB images, and mask write failures in `tools/generate_panoptic_sam3_segmentation_masks.py`.
- [x] 3.3 Implement optional `--continue-on-error` behavior in `tools/generate_panoptic_sam3_segmentation_masks.py` with per-item failure recording and continued processing of later images.
- [x] 3.4 Add progress reporting and a final run summary in `tools/generate_panoptic_sam3_segmentation_masks.py` covering processed, skipped, and failed images with sequence/camera/image-path metadata for failures.

## 4. Validation And Documentation

- [x] 4.1 Add documentation in `docs/` describing the script purpose, required SAM3 dependencies/checkpoints, canonical Panoptic input layout, output layout, and filename mapping rules.
- [x] 4.2 Validate strict-mode generation with a runnable command such as `uv run python tools/generate_panoptic_sam3_segmentation_masks.py --data-root /opt/data/panoptic_kinoptic_single_actor_cropped --sequences 161029_flute1 --cameras kinect_1 --segmentor-path /opt/data/SAM3_checkpoint`.
- [x] 4.3 Validate skip-existing and overwrite behavior with reruns on a small sequence-camera subset and confirm the written masks appear under `<sequence>/sam_segmentation_mask/<camera>/`.
- [x] 4.4 Validate `--continue-on-error` on a controlled failing case and confirm the run summary reports the skipped failure while later images continue processing.
