## ADDED Requirements

### Requirement: Config-Driven Input Modality/View Layout
The visualization pipeline SHALL build input panels and rerun entity paths from dataset configuration values instead of fixed modality/view assumptions. It SHALL read `modality_names` and each modality camera count (for example `rgb_cameras_per_sample`, `depth_cameras_per_sample`, `lidar_cameras_per_sample`) from the loaded config and create entities under `world/inputs/<modality>/view_<i>` for each configured view.

#### Scenario: Variable camera count is reflected in rerun layout
- **WHEN** a config defines `modality_names: ["rgb", "depth"]`, `rgb_cameras_per_sample: 4`, and `depth_cameras_per_sample: 2`
- **THEN** rerun input entities SHALL be created for `world/inputs/rgb/view_0..3` and `world/inputs/depth/view_0..1` with no extra hardcoded views

### Requirement: Shared Visualization Core for Multiple Inference Backends
The system SHALL provide a shared visualization core for sample loading, timeline stepping, and rerun logging that can be reused by multiple inference adapters. The shared core SHALL accept adapter outputs using explicit contracts for 2D keypoints and 3D keypoints with shapes `(T, V, J, 2)` and `(T, J, 3)` respectively.

#### Scenario: Existing MMHPE script uses shared core without CLI breakage
- **WHEN** `scripts/visualize_inference_rerun.py` is executed with existing required arguments
- **THEN** it SHALL run through the shared visualization core and produce rerun outputs with the same timeline semantics as before

### Requirement: Standardized Rerun Logging Namespaces
The visualization pipeline SHALL log inputs, outputs, and metadata using standardized rerun namespaces to allow direct cross-method comparison.

#### Scenario: Two scripts produce comparable rerun traces
- **WHEN** MMHPE visualization and SAM-3D-Body visualization are run on the same sample
- **THEN** both traces SHALL use the same namespace prefixes for inputs (`world/inputs/*`) and common output/info groups (`world/front/*`, `world/side/*`, `world/info/*`)
