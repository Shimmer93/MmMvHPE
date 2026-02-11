## ADDED Requirements

### Requirement: H36MMultiViewDataset can skip RGB loading
H36MMultiViewDataset SHALL provide a configuration toggle that disables RGB image loading
while still returning keypoints, camera parameters, and sample metadata.

#### Scenario: Keypoints-only training without RGB I/O
- **WHEN** the dataset is configured with image loading disabled
- **THEN** the dataset SHALL not read RGB image files from disk
- **AND** the sample SHALL still include keypoints and camera parameters needed for training

### Requirement: RGB-dependent transforms are not required when RGB loading is disabled
The H36MMultiViewDataset pipeline SHALL allow configurations that omit RGB transforms when
image loading is disabled.

#### Scenario: Pipeline without RGB transforms
- **WHEN** the dataset is configured with image loading disabled and a pipeline that omits RGB transforms
- **THEN** dataset initialization and iteration SHALL succeed without requiring RGB data
