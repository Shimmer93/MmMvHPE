# Model Pipeline

This document describes how MMHPE builds and runs model components during training/evaluation/inference.

## Core orchestration

`models/model_api.py` (`LitModel`) controls:
- modality backbones (`backbone_rgb`, `backbone_depth`, `backbone_lidar`, `backbone_mmwave`)
- feature aggregation (`aggregator`)
- task heads (`keypoint_head`, `smpl_head`, `camera_head`)
- metrics and logging

High-level flow:
1. Extract modality features from available inputs.
2. Aggregate modality features into fused tokens/features.
3. Compute losses in train step and predictions in val/test/predict steps.
4. Evaluate configured metrics on prediction dict.

## Module registry and discovery

Modules are instantiated by name through `misc.registry.create_model`.

To be constructible from config, classes must be exported via:
- `models/aggregators/__init__.py`
- `models/heads/__init__.py`
- `models/video_encoders/__init__.py`
- `models/pc_encoders/__init__.py`
- `models/__init__.py`

## Available module families

### Aggregators

- `XFiAggregator`
- `TransformerAggregator`
- `TransformerAggregatorV2`
- `TransformerAggregatorV2GlobalJoint`
- `TransformerAggregatorV2GlobalSMPL`
- `TransformerAggregatorV3`
- `TransformerAggregatorV3Lite`
- `TransformerAggregatorV4`
- `SimpleAggregator`
- `LEIRAggregator`

### Heads

- keypoint heads: `RegressionKeypointHead*`, `XFiRegressionHead`
- SMPL heads: `SMPLHead*`, `SMPLTokenHead*`, `VIBETokenHead*`
- camera heads: `VGGTCameraHead*`, `HeuristicCameraHead`, `KeypointCameraHeadV5`, `KeypointCameraGCNHeadV5`
- LEIR head: `LEIRHead`

## KeypointCameraGCNHeadV5 branch control

`KeypointCameraGCNHeadV5` supports selecting which branch contributes training loss:
- `train_branch: both` (default): use 2D modalities (`rgb`, `depth`) and 3D modalities (`lidar`, `mmwave`).
- `train_branch: 2d`: only `rgb`/`depth` camera and projection losses are applied.
- `train_branch: 3d`: only `lidar`/`mmwave` camera and projection losses are applied.

Notes:
- this only affects loss computation; prediction output shape/format stays unchanged.
- if a disabled-branch modality exists in `modalities`, its loss terms are skipped.

Example:

```yaml
camera_head:
  name: KeypointCameraGCNHeadV5
  params:
    train_branch: "3d"
```

## Notable runtime behaviors

- `camera_only=True` skips backbone+aggregator path and only runs camera head logic.
- if a backbone has `has_temporal: false`, `LitModel` flattens `(B, T, ...) -> (B*T, ...)`, applies encoder, then restores temporal shape.
- camera head API compatibility is handled with multi-signature fallback (`try/except TypeError`) for older/newer head implementations.
- optional `save_test_preds` gathers predictions from all distributed ranks and writes one merged pickle.

## Prediction dictionary conventions

Common keys produced by heads/steps:
- `pred_keypoints`
- `pred_smpl_params`
- `pred_smpl_keypoints`
- `pred_cameras`

Metrics read these keys plus GT fields from batch.

## Add a new model component

1. Implement class in appropriate folder (for example `models/heads/my_head.py`).
2. Export it in the package `__init__.py`.
3. Reference it from config.

Example:

```python
# models/heads/__init__.py
from .my_head import MyHead

__all__ = [
    # ...
    "MyHead",
]
```

```yaml
keypoint_head:
  name: MyHead
  params:
    hidden_dim: 512
```

## Common failure points

- class exists but not exported in `__init__.py` -> import/attribute errors from registry.
- mismatch between active modalities and aggregator/head expectations.
- mixed list/tensor batch values from dataset collate path causing downstream shape assumptions to fail.
- loading checkpoint with incompatible head names/keys (partially handled via selective loading in `LitModel`).
