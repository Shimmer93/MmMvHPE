# Config Guide

This project is config-driven. `main.py` loads one YAML, builds data/model modules, and runs Lightning train/test/predict loops.

## Runtime flow

1. CLI args are parsed in `main.py`.
2. YAML is loaded by `misc.utils.load_cfg`.
3. `misc.utils.merge_args_cfg` merges CLI and YAML.
4. `datasets.data_api.LitDataModule` and `models.model_api.LitModel` are built from merged args.

## Important precedence rule

`merge_args_cfg(args, cfg)` uses:

```python
dict = {**vars(args), **vars(cfg)}
```

This means YAML values override CLI values for overlapping keys.

Practical impact:
- `-b/--batch_size`, `-g/--gpus`, `--precision`, etc. only work from CLI if those keys are not already defined in YAML.
- if a run behaves unexpectedly, check YAML first.

## Minimal YAML shape

```yaml
strategy: ddp
precision: 16
epochs: 20

train_dataset:
  name: HummanPreprocessedDatasetV2
  params: { ... }
val_dataset:
  name: HummanPreprocessedDatasetV2
  params: { ... }
test_dataset:
  name: HummanPreprocessedDatasetV2
  params: { ... }

optim_name: AdamW
optim_params:
  lr: 0.0001

sched_name: LinearWarmupCosineAnnealingLR
sched_params:
  warmup_epochs: 4
  max_epochs: 20

backbone_rgb:
  name: TimmWrapper
  params: { model_name: vit_small_patch16_dinov3, pretrained: true }
  has_temporal: false

aggregator:
  name: TransformerAggregatorV4
  params: { ... }

keypoint_head:
  name: RegressionKeypointHeadV5
  params: { ... }

metrics:
  - name: MPJPE
    params: { affix: null }

train_pipeline:
  - name: VideoResize
    params: { size: [180, 320], keys: [input_rgb] }
  - name: ToTensor
    params: null
val_pipeline: *train_pipeline
test_pipeline: *train_pipeline
```

## Dataset blocks per stage

`LitDataModule.setup(stage)` expects:
- train: `train_dataset`, `val_dataset`
- test: `test_dataset`
- predict: `predict_dataset`

If you run `--predict`, include `predict_dataset` and `predict_pipeline` in YAML.

## Model/loss/metric instantiation

All modules are looked up by string name through `misc.registry`:
- model: from `models`
- loss: from `losses` (fallback `torch.nn`)
- metric: from `metrics`
- transform: from `datasets.transforms`

If a class is not exported in package `__init__.py`, config instantiation fails.

## Common run commands

### Train

```bash
uv run python main.py -c configs/dev/humman_smpl_token_v4.yml \
  -g 2 -n 1 -w 8 -b 32 -e 32 -p 2 --pin_memory \
  --exp_name dev_humman --version train_$(date +'%Y%m%d_%H%M%S')
```

### Test

```bash
uv run python main.py -c configs/dev/humman_smpl_token_v4.yml \
  -g 2 -n 1 -w 8 -b 32 -e 32 -p 2 --pin_memory \
  --exp_name dev_humman --version test_run --test \
  --checkpoint_path logs/dev_humman/<version>/<ckpt>.ckpt
```

### Predict

```bash
uv run python main.py -c configs/dev/humman_smpl_token_v4.yml \
  -g 1 -n 1 -w 8 -b 32 -e 32 -p 2 --pin_memory \
  --exp_name dev_humman --version pred_run --predict \
  --checkpoint_path logs/dev_humman/<version>/<ckpt>.ckpt
```

## Troubleshooting checklist

- config path exists and is for the same dataset/model family you intend to run.
- dataset class name in YAML is exported by `datasets/__init__.py`.
- all module names in YAML are exported by `models/__init__.py`, `losses/__init__.py`, `metrics/__init__.py`.
- run mode and dataset keys match (`--predict` needs `predict_dataset`).
- CUDA visibility and `-g` value are aligned with available devices.
