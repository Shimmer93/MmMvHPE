# MMHPE Documentation

This folder is the primary onboarding and maintenance documentation for MMHPE.

The goal is practical:
- explain what each major part does,
- explain the implementation ideas at a high level,
- record details that are easy to misunderstand,
- provide command examples you can run directly.

## Document map

- `docs/config_guide.md`: how config-driven runs work (`main.py` + YAML).
- `docs/data_pipeline.md`: datasets, transforms, sample format, and split logic.
- `docs/model_pipeline.md`: model composition, heads, metrics, and extension points.
- `docs/tools.md`: preprocessing and helper scripts with command-line examples.
- `docs/rerun_visualization.md`: rerun inference visualization scripts and shared helper modules.
- `docs/eval_fixed_lidar_frame_algorithm.md`: paper-ready algorithm summary for fixed-sensor evaluation and robust multi-sensor fusion.

## Quickstart

### 1) Environment

```bash
uv sync
```

### 2) Train

```bash
uv run python main.py \
  -c configs/dev/humman_smpl_token_v4.yml \
  -g 2 -n 1 -w 8 -b 32 -e 32 -p 2 --pin_memory \
  --exp_name dev_humman --version exp_$(date +'%Y%m%d_%H%M%S')
```

### 3) Test

```bash
uv run python main.py \
  -c configs/dev/humman_smpl_token_v4.yml \
  -g 2 -n 1 -w 8 -b 32 -e 32 -p 2 --pin_memory \
  --exp_name dev_humman --version eval_run \
  --test --checkpoint_path logs/dev_humman/<version>/<ckpt>.ckpt
```

### 4) Predict

```bash
uv run python main.py \
  -c configs/dev/humman_smpl_token_v4.yml \
  -g 1 -n 1 -w 8 -b 32 -e 32 -p 2 --pin_memory \
  --exp_name dev_humman --version pred_run \
  --predict --checkpoint_path logs/dev_humman/<version>/<ckpt>.ckpt
```

Prediction artifacts are written under:
- `logs/<exp_name>/<version>/<model_name>_predictions.pt`
- `logs/<exp_name>/<version>/<model_name>_test_predictions.pkl` (when `save_test_preds` is enabled)

## Debug print switch

`main.py` filters `[DEBUG]` print lines unless debug mode is enabled:

```bash
uv run python main.py -c <config>.yml --debug ...
```

## Documentation maintenance rule

Keep source comments short and focused. Put higher-level explanations, tricky assumptions, and usage examples in `docs/`.
