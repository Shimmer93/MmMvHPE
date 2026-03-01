## ADDED Requirements

### Requirement: EFC pose feature extractor
The system SHALL provide an EFC (3‑layer FCN) pose feature extractor that maps
per‑joint embeddings to per‑pose features in the shape \(T \times V \times C_P\).

#### Scenario: Extract pose features from embedded joints
- **WHEN** the extractor receives embeddings shaped \(T \times V \times J \times C_J\)
- **THEN** it SHALL output pose features shaped \(T \times V \times C_P\)

### Requirement: Configurable extractor selection
The system SHALL allow selecting the EFC extractor via config for FusionFormer runs.

#### Scenario: Select EFC baseline in config
- **WHEN** the config specifies the EFC extractor
- **THEN** FusionFormer SHALL use the EFC module for pose feature extraction
