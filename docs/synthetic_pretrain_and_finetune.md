# Synthetic Pretrain And Finetune Workflow

This document describes how to use the exported synthetic COCO dataset for:
- stage-1 HPE pretraining
- stage-2 camera-head pretraining
- real-data finetuning on HuMMan or Panoptic
- final fixed-lidar-frame evaluation

The workflow assumes the exported synthetic roots already exist, for example:
- train: `/opt/data/coco/synthetic_data/v0a_train2017`
- val: `/opt/data/coco/synthetic_data/v0a_val2017`

The training-facing synthetic loader is:
- `datasets/synthetic_exported_training_dataset.py`

It reads the existing sample-centric synthetic export directories directly and does not require a second HuMMan-style or Panoptic-style on-disk dataset tree.

## Dataset Contract

The synthetic loader supports:
- `target_format: humman`
- `target_format: panoptic`

Common sample behavior:
- one synthetic sample directory maps to one dataset sample
- only `seq_len=1` is supported
- RGB is loaded from the original source image path stored in the synthetic sample manifest
- LiDAR is loaded from the synthetic point-cloud artifact
- camera dicts are returned in the same shape used by the existing training pipelines

Returned training keys:
- `sample_id`
- `modalities`
- `input_rgb`
- `input_lidar`
- `rgb_camera`
- `lidar_camera`
- `gt_keypoints`
- `gt_global_orient`
- `gt_pelvis`
- `gt_keypoints_2d_rgb`
- `gt_keypoints_lidar`
- `gt_keypoints_pc_centered_input_lidar`

HuMMan-style synthetic samples additionally return:
- `gt_smpl_params`

Panoptic-style synthetic samples return:
- `gt_smpl_params` as zeros to preserve compatibility with the current model/test code

## Config Layout

HuMMan synthetic-transfer configs:
- `configs/exp/synthetic_transfer/humman/synthetic_stage1_pretrain.yml`
- `configs/exp/synthetic_transfer/humman/synthetic_stage2_pretrain.yml`
- `configs/exp/synthetic_transfer/humman/real_stage1_finetune_from_synth.yml`
- `configs/exp/synthetic_transfer/humman/real_stage2_finetune_from_synth.yml`
- `configs/exp/synthetic_transfer/humman/final_eval_from_synth.yml`

Panoptic synthetic-transfer configs:
- `configs/exp/synthetic_transfer/panoptic/synthetic_stage1_pretrain.yml`
- `configs/exp/synthetic_transfer/panoptic/synthetic_stage2_pretrain.yml`
- `configs/exp/synthetic_transfer/panoptic/real_stage1_finetune_from_synth.yml`
- `configs/exp/synthetic_transfer/panoptic/real_stage2_finetune_from_synth.yml`
- `configs/exp/synthetic_transfer/panoptic/final_eval_from_synth.yml`
- `configs/exp/synthetic_transfer/panoptic/final_eval_occluded_from_synth.yml`
- `configs/exp/synthetic_transfer/panoptic/final_eval_unoccluded_from_synth.yml`

## Example Commands

Sequential synthetic pretraining across both stages and both target formats:

```bash
GPUS=2 BATCH_SIZE=32 BATCH_SIZE_EVA=32 USE_WANDB=1 \
  bash scripts/run_synthetic_pretraining.sh
```

The runner:
- waits until all visible GPUs are under configurable memory/utilization thresholds
- runs HuMMan stage-1, HuMMan stage-2, Panoptic stage-1, then Panoptic stage-2
- automatically finds the best stage-1 checkpoint for each target format and passes it into stage-2
- writes per-stage launcher logs under `logs/synthetic_pretraining_runs/<run_id>/`

HuMMan stage-1 synthetic pretraining:

```bash
uv run python main.py \
  --cfg configs/exp/synthetic_transfer/humman/synthetic_stage1_pretrain.yml \
  --exp_name synthetic_transfer_humman_stage1 \
  --version pretrain_run1 \
  -g 1 -w 8 -b 16 -e 16 --pin_memory
```

HuMMan stage-2 synthetic pretraining:

```bash
uv run python main.py \
  --cfg configs/exp/synthetic_transfer/humman/synthetic_stage2_pretrain.yml \
  --exp_name synthetic_transfer_humman_stage2 \
  --version pretrain_run1 \
  --checkpoint_path <synthetic_stage1_ckpt> \
  -g 1 -w 8 -b 16 -e 16 --pin_memory
```

HuMMan real-data stage-1 finetuning:

```bash
uv run python main.py \
  --cfg configs/exp/synthetic_transfer/humman/real_stage1_finetune_from_synth.yml \
  --exp_name synthetic_transfer_humman_stage1 \
  --version finetune_run1 \
  --checkpoint_path <synthetic_stage1_ckpt> \
  -g 1 -w 8 -b 16 -e 16 --pin_memory
```

HuMMan real-data stage-2 finetuning:

```bash
uv run python main.py \
  --cfg configs/exp/synthetic_transfer/humman/real_stage2_finetune_from_synth.yml \
  --exp_name synthetic_transfer_humman_stage2 \
  --version finetune_run1 \
  --checkpoint_path <real_stage1_finetuned_ckpt> \
  -g 1 -w 8 -b 16 -e 16 --pin_memory
```

HuMMan prediction dump for final fixed-lidar-frame evaluation:

```bash
uv run python main.py \
  --cfg configs/exp/synthetic_transfer/humman/final_eval_from_synth.yml \
  --exp_name synthetic_transfer_humman_eval \
  --version final_eval \
  --checkpoint_path <real_stage1_finetuned_ckpt> \
  --test -g 1 -w 8 -e 16 --pin_memory
```

Then evaluate the dumped predictions:

```bash
uv run python tools/eval_fixed_lidar_frame.py \
  --pred-file logs/synthetic_transfer_humman_eval/final_eval/HummanVIBEToken_test_predictions.pkl \
  --projection-mode seq_lidar_ref
```

Panoptic stage-1 synthetic pretraining:

```bash
uv run python main.py \
  --cfg configs/exp/synthetic_transfer/panoptic/synthetic_stage1_pretrain.yml \
  --exp_name synthetic_transfer_panoptic_stage1 \
  --version pretrain_run1 \
  -g 1 -w 8 -b 16 -e 16 --pin_memory
```

Panoptic stage-2 synthetic pretraining:

```bash
uv run python main.py \
  --cfg configs/exp/synthetic_transfer/panoptic/synthetic_stage2_pretrain.yml \
  --exp_name synthetic_transfer_panoptic_stage2 \
  --version pretrain_run1 \
  --checkpoint_path <synthetic_stage1_ckpt> \
  -g 1 -w 8 -b 16 -e 16 --pin_memory
```

Panoptic real-data stage-1 finetuning:

```bash
uv run python main.py \
  --cfg configs/exp/synthetic_transfer/panoptic/real_stage1_finetune_from_synth.yml \
  --exp_name synthetic_transfer_panoptic_stage1 \
  --version finetune_run1 \
  --checkpoint_path <synthetic_stage1_ckpt> \
  -g 1 -w 8 -b 16 -e 16 --pin_memory
```

Panoptic real-data stage-2 finetuning:

```bash
uv run python main.py \
  --cfg configs/exp/synthetic_transfer/panoptic/real_stage2_finetune_from_synth.yml \
  --exp_name synthetic_transfer_panoptic_stage2 \
  --version finetune_run1 \
  --checkpoint_path <real_stage1_finetuned_ckpt> \
  -g 1 -w 8 -b 16 -e 16 --pin_memory
```

Panoptic prediction dump for fixed-lidar-frame evaluation:

```bash
uv run python main.py \
  --cfg configs/exp/synthetic_transfer/panoptic/final_eval_from_synth.yml \
  --exp_name synthetic_transfer_panoptic_eval \
  --version final_eval \
  --checkpoint_path <real_stage1_finetuned_ckpt> \
  --test -g 1 -w 8 -e 16 --pin_memory
```

Then evaluate the dumped predictions:

```bash
uv run python tools/eval_fixed_lidar_frame.py \
  --pred-file logs/synthetic_transfer_panoptic_eval/final_eval/PanopticHPE_test_predictions.pkl \
  --projection-mode seq_lidar_ref
```

## Notes

- The synthetic stage-2 configs use the dataset-provided GT 2D RGB and LiDAR-side 3D supervision directly; they do not need external JSON skeleton prediction files.
- The real HuMMan stage-2 configs still follow the current JSON-based workflow because they are intended to stay compatible with the existing real-data training setup.
- The Panoptic synthetic dataset reconstructs LiDAR-side supervision on the fly from the base synthetic sample artifacts to avoid duplicating files under `exports/panoptic`.
