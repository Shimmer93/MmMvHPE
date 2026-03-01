## MODIFIED Requirements

### Requirement: Timeline Logging Per Visualized Frame
The script SHALL log each visualized frame as a separate rerun timeline step while preserving standard namespace paths. For GT logging, the script SHALL apply one selected GT coordinate space consistently across all visualized frames (`canonical` or `camera`) and across all selected views.

#### Scenario: Multiple frames generate multiple timeline steps
- **WHEN** `--num-frames 5` is used
- **THEN** rerun output SHALL contain five sequential timeline steps with frame data under `world/inputs/*`, `world/front/*`, `world/side/*`, and metadata under `world/info/*`

#### Scenario: Camera GT mode stays consistent across multiframe window
- **WHEN** GT coordinate space is `camera` and `--num-frames` is greater than 1
- **THEN** each frame's GT keypoints SHALL be transformed using that frame/view camera extrinsics before timeline logging

#### Scenario: Canonical GT mode stays consistent across multiframe window
- **WHEN** GT coordinate space is `canonical` and `--num-frames` is greater than 1
- **THEN** each frame's GT keypoints SHALL be logged in canonical space without per-frame camera transform
