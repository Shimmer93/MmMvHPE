## MODIFIED Requirements

### Requirement: Ground-Truth Keypoint Logging for SAM-3D-Body Rerun
The SAM-3D-Body rerun visualization SHALL log GT keypoints when GT keypoint annotations are present in the selected dataset sample. The script SHALL support logging GT in one selected coordinate space per run (`canonical` or `camera`) and SHALL NOT mix both spaces in the same run output.

#### Scenario: GT keypoints are available in sample
- **WHEN** a sample contains `gt_keypoints` with 3D joint coordinates
- **THEN** the script SHALL log GT skeleton entities under standard namespaces (`world/front/ground_truth/*`, `world/side/ground_truth/*`)

#### Scenario: Camera coordinate GT is requested
- **WHEN** GT coordinate space is set to `camera` and `rgb_camera` extrinsics are available for the selected view
- **THEN** the script SHALL transform GT keypoints to camera coordinates before logging ground-truth skeleton entities

#### Scenario: Canonical coordinate GT is requested
- **WHEN** GT coordinate space is set to `canonical`
- **THEN** the script SHALL log GT keypoints without camera-frame transform

### Requirement: Explicit GT Availability Metadata
The SAM-3D-Body rerun visualization SHALL emit metadata indicating whether GT keypoints and GT mesh were logged for the sample. The metadata SHALL also include which GT coordinate space was used.

#### Scenario: Missing GT mesh is handled without failure
- **WHEN** GT mesh fields are absent but prediction outputs are available
- **THEN** the script SHALL continue execution, log available GT/prediction data, and write GT availability status under `world/info/*`

#### Scenario: GT coordinate-space metadata is present
- **WHEN** GT keypoints are logged
- **THEN** the script SHALL log `gt_coordinate_space` metadata with value `canonical` or `camera`
