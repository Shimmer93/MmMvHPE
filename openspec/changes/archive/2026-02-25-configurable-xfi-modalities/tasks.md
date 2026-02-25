## 1. Aggregator Modality Contract

- [x] 1.1 Update `models/aggregators/xfi_aggregator.py` to accept `active_modalities` in `XFiAggregator` params and validate allowed modality names (`rgb`, `depth`, `mmwave`, `lidar`).
- [x] 1.2 Enforce canonical modality ordering (`rgb`, `depth`, `mmwave`, `lidar`) internally before branch mapping in `linear_projector` and `XFiAggregator.forward`.
- [x] 1.3 Refactor feature-to-branch mapping so absent modalities are skipped safely and never routed to wrong projector branches.
- [x] 1.4 Add fail-fast checks in aggregator forward for missing required modality features and incompatible feature shapes.

## 2. Config Updates

- [x] 2.1 Update `configs/baseline/unseen_view_generalization/humman_xfi.yml` to explicitly set `aggregator.params.active_modalities` for intended modality sets.
- [x] 2.2 Verify `configs/baseline/unseen_view_generalization/humman_xfi.yml` declares depth-to-lidar modality intent unambiguously (for example RGB+LiDAR).

## 3. Validation

- [x] 3.1 Run a smoke test for RGB+depth XFi path via synthetic aggregator forward and confirm pass through projection/fusion.
- [x] 3.2 Run a smoke test for RGB+LiDAR (depth converted to lidar) with `uv run python main.py ...` and confirm no `NoneType` projector errors.
- [x] 3.3 Add or run targeted checks that assert invalid `active_modalities`, missing required modality feature, and invalid feature shape fail with explicit errors.

## 4. Documentation

- [x] 4.1 Add `docs/` guidance describing XFi modality configuration, canonical ordering, and fail-fast validation behavior.
- [x] 4.2 Include concrete command examples for running RGB+depth and RGB+LiDAR XFi experiments using `uv run`.
