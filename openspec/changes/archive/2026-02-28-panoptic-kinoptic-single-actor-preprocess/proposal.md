## Why

We need a dedicated preprocessing pipeline for single-actor CMU Panoptic Kinoptic data so training/evaluation can load data faster from compact preprocessed outputs instead of large raw video/depth sources.

Current tooling in `tools/data_preprocess.py` covers MMFi/HuMMan/H36M but does not support Panoptic Kinoptic. The Panoptic single-actor pipeline also has specific requirements:
- per-sequence structure must be preserved,
- RGB crops should be generated in a HuMMan-cropped style,
- multi-sensor data must be synchronized,
- users must be able to preprocess a selected subset of sequences because downloads may be incomplete.

## What Changes

- Add a new Panoptic Kinoptic preprocessing script under `tools/` for **single-actor** sequences.
- The script will operate on raw Panoptic sequence folders (for example under `/data/shared/panoptic-toolbox/<sequence>`), and produce compact preprocessed outputs.
- Synchronization will be explicit and sequence-local using released sync metadata (no cross-sequence mixing).
- Output layout will retain sequence grouping: each sequence stays under its own output folder.
- The CLI will support selecting specific sequences via args/file list to handle partial/in-progress downloads.
- The initial validation target is `161029_piano2`.

Scope boundaries and non-goals:
- In scope: single-actor Kinoptic preprocessing for faster loading and sequence-safe outputs.
- Out of scope: multi-actor support, full Panoptic (non-kinoptic) preprocessing, changing model architectures, or broad dataset class refactors in this step.

## Capabilities

### New Capabilities
- `panoptic-kinoptic-single-actor-preprocess-script`: Preprocess selected single-actor Kinoptic sequences into compact per-sequence outputs.
- `panoptic-kinoptic-sequence-synchronized-crop-format`: Generate synchronized, cropped per-frame outputs suitable for fast training-time loading while preserving sequence boundaries.

### Modified Capabilities
- None.

## Impact

- Affected code:
  - new preprocessing script under `tools/` (Panoptic Kinoptic specific),
  - documentation updates under `docs/` for usage and output format,
  - OpenSpec updates for proposal/spec/tasks.
- Affected modalities/components:
  - RGB (cropped output),
  - synchronization metadata usage (Kinoptic sync tables),
  - sequence-level GT body annotations (`hdPose3d_stage1_coco19`).
- Workflow impact:
  - enables selective preprocessing of downloaded single-actor sequences,
  - reduces I/O and decode overhead during training by materializing compact outputs,
  - keeps output compatibility with sequence-aware downstream usage.
