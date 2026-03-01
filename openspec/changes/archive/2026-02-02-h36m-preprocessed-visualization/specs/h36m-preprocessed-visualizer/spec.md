## ADDED Requirements

### Requirement: Config-driven multiview overlay visualization
The visualization script SHALL load a specified config file and split to construct the
H36MMultiViewDataset sample, then overlay 2D keypoints on each RGB view.

#### Scenario: Render multiview 2D overlays
- **WHEN** the script is invoked with a config path and split
- **THEN** it SHALL render one RGB image per view with 2D keypoints overlaid
- **AND** it SHALL use the 2D keypoints provided by the dataset pipeline

### Requirement: Render GT 3D skeleton for the same sample
The visualization script SHALL display the GT 3D skeleton for the same sample used in the
2D overlay to allow cross-checking of projection correctness.

#### Scenario: Render GT 3D skeleton alongside overlays
- **WHEN** the script renders the multiview overlays
- **THEN** it SHALL render the GT 3D skeleton for the same sample in the output figure

### Requirement: CLI selection of split and sample
The visualization script SHALL support CLI arguments to select dataset split and sample
index or number.

#### Scenario: Select a specific sample for inspection
- **WHEN** the user passes a split and sample index via CLI
- **THEN** the script SHALL visualize that specific sample from the dataset
