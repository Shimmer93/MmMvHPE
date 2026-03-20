## Why

The synthetic SAM-3D-Body pipeline now exports HuMMan-style and Panoptic-style supervision bundles, but MMHPE still has no training-facing dataset class or staged configs that can consume those exports directly. Without that integration, it is not possible to run the intended experiment of pretraining on synthetic RGB+LiDAR samples and then finetuning on HuMMan or Panoptic with the existing two-stage HPE and camera-head workflow.

## What Changes

- Add a synthetic exported-dataset class under `datasets/` that reads per-sample `exports/` bundles and exposes the same training-facing sample contract used by the current HuMMan and Panoptic pipelines.
- Support both HuMMan-style SMPL24 supervision and Panoptic-style joints19 supervision from the exported synthetic roots, including RGB, LiDAR, GT cameras, 2D RGB keypoints, and LiDAR-centered keypoints required by camera-head training.
- Add config sets for synthetic pretraining and real-data finetuning for both HuMMan-style and Panoptic-style pipelines.
- Add final-evaluation configs aligned with the current fixed-lidar-frame workflow so stage-2 results can be measured through `tools/eval_fixed_lidar_frame.py` instead of relying on `main.py` metrics alone.
- Document how to run stage-1 pretraining, stage-1 finetuning, stage-2 camera-head training, and final evaluation for both styles.

## Capabilities

### New Capabilities
- `synthetic-exported-training-dataset`: Load exported synthetic samples for training and evaluation with HuMMan-style or Panoptic-style supervision contracts.
- `synthetic-pretrain-and-finetune-configs`: Provide staged pretrain, finetune, and final-eval configs for synthetic-to-real training in both HuMMan-style and Panoptic-style pipelines.

### Modified Capabilities
- None.

## Impact

- Affected code: `datasets/`, dataset registration in `datasets/__init__.py`, dataset docs in `docs/`, and new YAML configs under `configs/`.
- Affected workflow: RGB + LiDAR staged training, including stage-1 HPE pretraining/finetuning and stage-2 camera-head training/evaluation.
- Runtime outputs: synthetic pretrain/finetune checkpoints and prediction dumps under `logs/`, plus fixed-lidar-frame evaluation outputs produced by `tools/eval_fixed_lidar_frame.py`.
