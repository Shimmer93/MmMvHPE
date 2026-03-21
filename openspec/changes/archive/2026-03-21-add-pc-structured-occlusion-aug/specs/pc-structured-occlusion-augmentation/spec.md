## ADDED Requirements

### Requirement: The Training Pipeline Shall Support Runtime Structured Point-Cloud Occlusion Augmentation
The system SHALL provide a point-cloud training transform that applies structured occlusion or dropout to `input_lidar` at runtime without requiring any changes to the saved dataset artifacts.

#### Scenario: Apply structured occlusion during training
- **WHEN** a training config enables the structured occlusion transform in the point-cloud pipeline
- **THEN** the dataset pipeline SHALL modify `input_lidar` according to the configured augmentation policy before the sample is consumed by the model

### Requirement: Structured Occlusion Augmentation Shall Use Range-Image Blob Dropout
The transform SHALL support range-image blob dropout by projecting `input_lidar` into a range-image-style view, masking one or more contiguous blobs in that 2D representation, and removing the corresponding 3D points.

#### Scenario: Remove a contiguous range-image region
- **WHEN** the transform is configured with range-image blob dropout and applied to a LiDAR sample
- **THEN** it SHALL remove a contiguous region of points according to the configured blob parameters

### Requirement: Structured Occlusion Augmentation Shall Be Configurable From YAML
The augmentation SHALL be configurable from the existing YAML pipeline system, including whether it is enabled, the probability of application, and range-image blob parameters such as blob count, blob size, and optional shape or jitter controls.

#### Scenario: Toggle augmentation strength from config
- **WHEN** two training configs enable the same structured occlusion transform with different probabilities or severity values
- **THEN** the runtime pipeline SHALL apply the augmentation according to each config without requiring code edits

### Requirement: Structured Occlusion Augmentation Shall Preserve Downstream Pipeline Compatibility
The transform SHALL preserve the tensor/data contract expected by the downstream point-cloud pipeline and SHALL be designed to run before downstream point-cloud centering and padding transforms.

#### Scenario: Use structured occlusion before PC centering
- **WHEN** the transform is inserted into an existing LiDAR training pipeline before `PCCenterWithKeypoints`
- **THEN** the sample SHALL remain valid for the rest of the configured pipeline and SHALL NOT break batch collation because of the augmentation

### Requirement: The Repository Shall Document Structured Occlusion Usage For Synthetic Transfer Experiments
The repository SHALL provide documentation and example config usage for enabling the augmentation in synthetic pretraining or synthetic-to-real finetuning experiments.

#### Scenario: Enable augmentation from documented examples
- **WHEN** a user follows the documented structured occlusion example for a synthetic-transfer config
- **THEN** they SHALL be able to enable the runtime augmentation without inventing new config syntax or dataset artifacts
