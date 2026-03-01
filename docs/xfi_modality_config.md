# XFi Modality Configuration

This document defines how to configure modality usage for `XFiAggregator`.

## Canonical Modality Order

XFi uses this canonical order internally:

- `rgb`
- `depth`
- `mmwave`
- `lidar`

`active_modalities` can be provided in any order in config, but the aggregator always canonicalizes it before branch mapping.

## Required Config

Set `aggregator.params.active_modalities` in the experiment config.

Example (`configs/baseline/unseen_view_generalization/humman_xfi.yml`):

```yaml
aggregator:
  name: 'XFiAggregator'
  params:
    active_modalities: ['rgb', 'lidar']
    input_dim: 512
    output_dim: 512
    num_modalities: 4
    dim_expansion: 2
    hidden_dim: 512
    num_heads: 8
    dim_heads: 64
    model_depth: 4
```

## Validation Behavior (Fail-Fast)

`XFiAggregator` raises explicit errors when:

- `active_modalities` contains unsupported or duplicate modality names.
- A configured modality is missing its runtime feature tensor in aggregator forward.
- Feature tensor shape is incompatible with expected modality shape:
  - `rgb` / `depth`: `[B, T, C, H, W]`
  - `mmwave` / `lidar` features: `[B, T, N, C]`
  - `input_lidar` (for positional encoding): `[B, T, N, 3]` when `lidar` is active.

If `active_modalities` is omitted, the aggregator infers modalities from non-`None` features and emits a one-time warning.

## Depth-to-LiDAR Intent

If dataset config uses:

- `modality_names: ['rgb', 'depth']`
- `convert_depth_to_lidar: true`

then the runtime point-cloud branch is LiDAR-style input (`input_lidar`). In this case, set:

- `active_modalities: ['rgb', 'lidar']`

## Run Examples

RGB + LiDAR (depth converted to LiDAR) from baseline config:

```bash
uv run python main.py \
  -c configs/baseline/unseen_view_generalization/humman_xfi.yml \
  -g 1 -n 1 -w 8 -b 16 -e 16 \
  --exp_name baseline --version xfi_rgb_lidar --wandb_offline
```

RGB + Depth example (requires an XFi config that keeps depth features active):

```bash
uv run python main.py \
  -c <your_rgb_depth_xfi_config>.yml \
  -g 1 -n 1 -w 8 -b 16 -e 16 \
  --exp_name baseline --version xfi_rgb_depth --wandb_offline
```
