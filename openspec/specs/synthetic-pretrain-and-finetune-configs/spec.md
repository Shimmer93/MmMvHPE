# synthetic-pretrain-and-finetune-configs Specification

## Purpose
TBD - created by archiving change add-synthetic-pretrain-dataset-configs. Update Purpose after archive.

## Requirements
### Requirement: The Repository Shall Provide Synthetic Stage-1 Pretrain Configs
The system SHALL provide stage-1 training configs that use the exported synthetic dataset for HuMMan-style and Panoptic-style RGB+LiDAR HPE pretraining.

#### Scenario: HuMMan-style synthetic stage-1 config
- **WHEN** a user loads the HuMMan-style synthetic stage-1 config
- **THEN** it SHALL use the synthetic exported dataset with `target_format: humman`, `modality_names: ['rgb', 'lidar']`, and a HuMMan-style model with SMPL supervision

#### Scenario: Panoptic-style synthetic stage-1 config
- **WHEN** a user loads the Panoptic-style synthetic stage-1 config
- **THEN** it SHALL use the synthetic exported dataset with `target_format: panoptic`, `modality_names: ['rgb', 'lidar']`, and a Panoptic-style keypoint-only model with joints19 supervision

### Requirement: The Repository Shall Provide Synthetic Stage-2 Camera-Head Pretrain Configs
The system SHALL provide stage-2 camera-head pretraining configs for both HuMMan-style and Panoptic-style synthetic training.

#### Scenario: Synthetic stage-2 config uses direct synthetic supervision
- **WHEN** a user loads a synthetic stage-2 config
- **THEN** it SHALL consume `gt_keypoints_2d_rgb`, LiDAR-side 3D skeleton supervision, and GT camera targets directly from the synthetic dataset without requiring external JSON prediction files

### Requirement: The Repository Shall Provide Synthetic-To-Real Finetune Configs
The system SHALL provide real-data finetune configs for HuMMan and Panoptic that initialize from synthetic-pretrained checkpoints while preserving the existing real-dataset workflow.

#### Scenario: Real HuMMan finetune from synthetic checkpoint
- **WHEN** a user loads the HuMMan finetune config
- **THEN** it SHALL keep the HuMMan real-dataset loader and expose explicit checkpoint fields for initializing stage-1 or stage-2 from the synthetic-pretrained runs

#### Scenario: Real Panoptic finetune from synthetic checkpoint
- **WHEN** a user loads the Panoptic finetune config
- **THEN** it SHALL keep the Panoptic real-dataset loader and expose explicit checkpoint fields for initializing stage-1 or stage-2 from the synthetic-pretrained runs

### Requirement: The Repository Shall Provide Final Evaluation Configs For Fixed-Lidar-Frame Evaluation
The system SHALL provide evaluation configs for HuMMan and Panoptic synthetic-to-real experiments that align with the existing fixed-lidar-frame workflow.

#### Scenario: Final evaluation config carries stage-1 and stage-2 checkpoint references
- **WHEN** a user loads a final evaluation config for HuMMan or Panoptic
- **THEN** it SHALL include the stage-1 checkpoint path and stage-2 `pretrained_camera_head_path` needed for prediction dumping and follow-up evaluation through `tools/eval_fixed_lidar_frame.py`

#### Scenario: Documentation includes final evaluation commands
- **WHEN** the synthetic-to-real workflow docs are updated
- **THEN** they SHALL include concrete commands for stage-1 pretraining, stage-1 finetuning, stage-2 camera-head training, prediction dumping, and `tools/eval_fixed_lidar_frame.py`
