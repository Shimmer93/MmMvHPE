## ADDED Requirements

### Requirement: Sequence-Local Panoptic RGB Traversal
The system SHALL provide a standalone CLI script under `tools/` that traverses Panoptic RGB images directly from the dataset filesystem without requiring a training or evaluation config.

The script SHALL accept a dataset root containing sequence directories and SHALL treat the canonical RGB input layout for a sequence `S` as:
- `<dataset_root>/S/rgb/<camera_name>/`

The script SHALL support filtering by explicit sequence names and explicit camera names so partially downloaded datasets can be processed incrementally. If a requested sequence directory, RGB root, or camera directory is missing, the script MUST fail with an explicit error unless optional continue-on-error mode is enabled.

#### Scenario: Traverse one selected sequence camera
- **WHEN** the user runs the script with dataset root `R`, sequence `161029_flute1`, and camera `kinect_1`
- **THEN** the script SHALL read RGB images from `R/161029_flute1/rgb/kinect_1/`
- **AND** it SHALL process only that sequence-camera stream

#### Scenario: Missing RGB camera directory in strict mode
- **WHEN** the user requests sequence `161029_flute1` and camera `kinect_3` but `<dataset_root>/161029_flute1/rgb/kinect_3/` does not exist
- **THEN** the script SHALL terminate with an explicit error naming the missing path

### Requirement: SAM3DBody-Compatible Person Mask Generation
The script SHALL generate segmentation masks using the same SAM3 segmentation path used by the repository’s SAM-3D-Body integration. The prompt SHALL be `person`.

For each RGB image, the script SHALL obtain all confident person masks returned by the SAM3 path and SHALL union them into a single binary foreground mask. The script MUST NOT silently keep only one person when multiple person masks are returned.

If SAM3 dependencies or checkpoints are unavailable, the script MUST fail fast with an actionable error before starting image traversal unless optional continue-on-error mode is enabled for per-sequence or per-camera failures after startup validation.

#### Scenario: Single person image
- **WHEN** SAM3 returns one confident person mask for an RGB image
- **THEN** the script SHALL write a binary mask representing that person region for the image

#### Scenario: Multiple person image
- **WHEN** SAM3 returns multiple confident person masks for an RGB image
- **THEN** the script SHALL compute the union of those masks
- **AND** it SHALL write one combined binary mask for that RGB image

### Requirement: Sequence-Camera Mask Output Layout
For each processed sequence and camera, the script SHALL write masks under:
- `<sequence_root>/sam_segmentation_mask/<camera_name>/`

The script SHALL produce one mask file per input RGB image. The output mask filename SHALL preserve the RGB image basename. If the RGB image extension is already lossless, the script SHALL keep the full filename. If the RGB image extension is lossy, the script SHALL write a `.png` mask using the same basename stem.

The written mask image MUST be binary-valued and lossless on disk.

#### Scenario: PNG RGB input
- **WHEN** the input RGB image is `000123.png`
- **THEN** the output mask path SHALL be `<sequence_root>/sam_segmentation_mask/<camera_name>/000123.png`

#### Scenario: JPEG RGB input
- **WHEN** the input RGB image is `000123.jpg`
- **THEN** the output mask path SHALL be `<sequence_root>/sam_segmentation_mask/<camera_name>/000123.png`
- **AND** the mask SHALL remain lossless and binary-valued

### Requirement: Incremental Execution And Overwrite Control
The script SHALL support incremental reruns by skipping already existing output masks by default. The script SHALL also provide an explicit overwrite mode that regenerates existing masks.

The script SHALL expose an optional continue-on-error mode. In strict mode, the first unrecoverable sequence, camera, image read, or mask write failure MUST stop the run. In continue-on-error mode, the script SHALL record the failure, skip the failing item, and continue processing later items.

#### Scenario: Skip existing mask by default
- **WHEN** the output mask file for an RGB image already exists and overwrite mode is disabled
- **THEN** the script SHALL leave the existing file unchanged
- **AND** it SHALL continue to the next RGB image

#### Scenario: Continue after one image failure
- **WHEN** continue-on-error mode is enabled and one RGB image cannot be decoded
- **THEN** the script SHALL record that image failure in its run summary
- **AND** it SHALL continue processing the remaining images in the selected sequence-camera stream

### Requirement: Progress And Run Summary Reporting
The script SHALL emit progress information suitable for long-running Panoptic exports and SHALL produce a run summary that records processed image counts, skipped image counts, and failure counts.

The run summary SHALL identify failures with enough metadata to locate the affected sequence, camera, and RGB image path. The script MAY print the summary to stdout or write it to a simple report file, but it MUST provide the summary at the end of the run.

#### Scenario: Successful partial batch summary
- **WHEN** the user processes multiple sequence-camera streams
- **THEN** the script SHALL report how many RGB images were processed, skipped as already existing, and failed
- **AND** the summary SHALL identify counts per run

#### Scenario: Continue-on-error failure summary
- **WHEN** continue-on-error mode is enabled and at least one sequence-camera item fails
- **THEN** the final summary SHALL list the failing sequence, camera, and image path entries that were skipped due to errors
