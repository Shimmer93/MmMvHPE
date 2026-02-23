## ADDED Requirements

### Requirement: Coordinate-Space Mode for MMHPE Rerun Inference Visualization
`scripts/visualize_inference_rerun.py` SHALL support two coordinate visualization modes: `world` and `sensor`. The default SHALL preserve existing behavior (`world`).

#### Scenario: Default invocation preserves existing output semantics
- **WHEN** the script is executed without coordinate-space arguments
- **THEN** it SHALL use `world` mode and produce the same coordinate interpretation as current runs

#### Scenario: Sensor mode is explicitly requested
- **WHEN** the script is executed with coordinate-space mode set to `sensor`
- **THEN** it SHALL switch to sensor-frame visualization behavior for 3D GT and prediction entities

### Requirement: Explicit Sensor Reference Selection in Sensor Mode
In `sensor` mode, the script SHALL require explicit reference-frame selection and SHALL reject runs that omit reference sensor/view parameters.

#### Scenario: Missing reference selection fails fast
- **WHEN** coordinate-space mode is `sensor` and reference sensor/view is not provided
- **THEN** the script SHALL terminate with an explicit error describing required selection arguments

#### Scenario: Supported sensor labels include lidar
- **WHEN** coordinate-space mode is `sensor` and user selects `lidar` as reference sensor label
- **THEN** the script SHALL treat it as the depth-derived point-cloud sensor frame used by current HuMMan pipeline

### Requirement: Consistent GT/Prediction Sensor-Frame Transform
In `sensor` mode, the script SHALL transform both GT and prediction 3D entities into the same selected sensor frame before logging. Transformation SHALL be applied consistently across keypoints and mesh vertices when those outputs are available.

#### Scenario: GT and prediction keypoints share one sensor frame
- **WHEN** sensor mode is enabled and both GT and predicted 3D keypoints are present
- **THEN** both SHALL be expressed in the selected sensor frame before rerun logging

#### Scenario: Temporal/multiview samples remain frame-consistent
- **WHEN** a sample includes temporal windows and/or multiple views
- **THEN** sensor-frame transform SHALL use the selected reference view deterministically for all logged frames

### Requirement: Sensor-Mode Metadata and View Coordinates
In `sensor` mode, rerun metadata SHALL include selected coordinate/reference settings and rerun view coordinates SHALL be camera-style Y-down.

#### Scenario: Metadata records sensor-frame context
- **WHEN** a sensor-mode run is recorded
- **THEN** metadata under `world/info/*` SHALL include coordinate mode, reference sensor label, and reference view index

#### Scenario: Sensor-mode view coordinates are Y-down
- **WHEN** sensor mode is active
- **THEN** the script SHALL log world/front/side view coordinates using `RIGHT_HAND_Y_DOWN`
