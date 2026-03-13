## ADDED Requirements

### Requirement: Panoptic runtime masking transform SHALL apply sequence-local SAM3 masks to configured modalities
The system SHALL provide one Panoptic-specific dataset transform that reads sequence-local SAM3 segmentation masks and applies them at runtime to loaded Panoptic sample modalities. The transform SHALL accept a configuration argument that selects one or both target modalities from `rgb` and `depth`. For RGB, the transform SHALL modify `input_rgb` directly in the RGB image plane. For depth, the transform SHALL modify `input_depth` by reprojecting the RGB-plane mask into the depth image plane through the calibrated RGB-depth geometry. Masked-out RGB pixels SHALL be set to zero in all channels, and masked-out depth pixels SHALL be set to numeric zero while preserving the input array dtype and shape.

#### Scenario: Apply mask to RGB only
- **WHEN** a Panoptic sample contains `input_rgb`, the transform is configured with `apply_to=[rgb]`, and the corresponding mask exists for the sample frame and selected RGB camera
- **THEN** the transform SHALL zero masked-out RGB pixels, SHALL leave RGB shape unchanged, and SHALL NOT modify `input_depth`

#### Scenario: Apply mask to both RGB and depth with one decoded mask per frame-camera pair
- **WHEN** a Panoptic sample contains both `input_rgb` and `input_depth`, the transform is configured with `apply_to=[rgb, depth]`, and the corresponding RGB-camera masks exist for the selected camera/frame pairs
- **THEN** the transform SHALL apply zero-filled masking to both modalities
- **AND** it SHALL use one mask decode per frame-camera pair even when both modalities are masked

#### Scenario: Reproject RGB mask into the depth image plane
- **WHEN** the transform is configured to mask `depth` and a depth frame is present
- **THEN** it SHALL back-project valid depth pixels with `K_depth`
- **AND** it SHALL transform those 3D points from depth camera coordinates to RGB camera coordinates using the calibrated camera geometry
- **AND** it SHALL project them with `K_color`
- **AND** it SHALL sample the RGB mask in the RGB image plane to decide which depth pixels remain foreground

#### Scenario: Ignore inactive target modality keys
- **WHEN** the transform is configured with `apply_to=[rgb, depth]` but the sample contains only one of `input_rgb` or `input_depth`
- **THEN** the transform SHALL apply masking only to the present configured modality and SHALL NOT raise an error solely because the other configured modality key is absent from the sample

### Requirement: Panoptic runtime masking transform SHALL resolve masks deterministically from sample metadata
The transform SHALL resolve every mask path from the Panoptic preprocessed dataset layout using sample metadata, not by directory scanning. The expected mask location SHALL be `<data_root>/<seq_name>/sam_segmentation_mask/<camera_name>/<frame_stem>.png`, where `seq_name` and selected camera names come from the sample and `frame_stem` corresponds to the synchronized body-frame id for the specific frame in the temporal window. The transform SHALL support the same camera naming convention used by the Panoptic dataset sample, including normalized Kinect camera names.

#### Scenario: Resolve single-frame single-view mask path
- **WHEN** a single-frame Panoptic sample provides one selected RGB camera and one synchronized body-frame id
- **THEN** the transform SHALL resolve exactly one mask path under that sequence's `sam_segmentation_mask/<camera>/` directory using the body-frame id stem and `.png` extension

#### Scenario: Resolve temporal multiview mask paths
- **WHEN** a Panoptic sample contains a temporal window and multiple selected views for a configured modality
- **THEN** the transform SHALL resolve one mask path per frame per selected camera using the synchronized frame ids already represented by the sample

### Requirement: Panoptic runtime masking transform SHALL fail fast on invalid mask inputs
The transform SHALL raise an explicit error when the expected mask file is missing, unreadable, spatially mismatched with the corresponding frame, or cannot be resolved from the sample metadata. The transform SHALL NOT silently skip masking, scan for alternate candidate files, or leave the frame unchanged in those cases.

#### Scenario: Missing mask file
- **WHEN** the transform expects a mask file for a configured modality frame-camera pair and that file does not exist
- **THEN** the transform SHALL raise an error that identifies the sequence, camera, frame, and expected mask path

#### Scenario: Unreadable or invalid mask image
- **WHEN** the resolved mask file exists but cannot be decoded as an image
- **THEN** the transform SHALL raise an error that identifies the resolved mask path

#### Scenario: Shape mismatch between mask and frame
- **WHEN** the resolved mask image height or width differs from the corresponding RGB or depth frame
- **THEN** the transform SHALL raise an error that identifies the frame shape, mask shape, and associated sequence-camera-frame identity

#### Scenario: Missing calibration metadata for depth reprojection
- **WHEN** the transform is configured to mask `depth` and the sample or sequence metadata cannot provide the RGB/depth intrinsics or the calibrated camera relation needed for reprojection
- **THEN** the transform SHALL raise an explicit error identifying the affected sequence and camera

### Requirement: Panoptic runtime masking transform SHALL remain compatible with config-driven Panoptic dataset pipelines
The transform SHALL be usable from existing YAML dataset pipelines without changes to `main.py`, `datasets/data_api.py`, or model APIs. The transform SHALL operate on raw loaded Panoptic image arrays before normalization, tensor formatting, or model-specific input formatting. Documentation SHALL include Panoptic config examples for RGB-only masking, depth-only masking, and combined RGB+depth masking.

#### Scenario: Use transform in a Panoptic config pipeline
- **WHEN** a Panoptic train, validation, test, or visualization config inserts the masking transform before normalization or formatting steps
- **THEN** the dataset pipeline SHALL produce masked samples with the same downstream sample structure expected by existing transforms and model code

#### Scenario: Remove transform from config
- **WHEN** a Panoptic config does not include the masking transform
- **THEN** dataset loading SHALL preserve the current unmasked behavior with no required changes to entrypoints or model definitions
