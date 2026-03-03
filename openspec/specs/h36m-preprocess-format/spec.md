# Capability: h36m-preprocess-format

## Purpose
Specify the output layout and storage formats for the preprocessed H36M dataset.

## Requirements

### Requirement: Output layout
The preprocessed H36M dataset SHALL be stored under a root directory with modality subfolders
(e.g., `rgb/`, `gt/`, and optional metadata) to enable fast loading.

#### Scenario: Layout check
- **WHEN** preprocessing completes
- **THEN** the output directory contains `rgb/` and `gt/` subfolders

### Requirement: Compact storage format
The RGB frames SHALL be stored as JPEG at 480x640; 3D joint arrays SHALL be stored in compact
NumPy format (float16) for SSD efficiency.

#### Scenario: Output formats
- **WHEN** inspecting a processed sample
- **THEN** RGB is a 480x640 JPEG and joints are stored as float16 arrays
