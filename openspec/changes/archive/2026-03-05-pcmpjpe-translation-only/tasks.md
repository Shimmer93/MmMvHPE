## 1. Metric behavior update

- [x] 1.1 Update `metrics/mpjpe.py` so `pcmpjpe_func` performs pelvis translation centering only, without root-orientation alignment.
- [x] 1.2 Update `metrics/smpl_metrics.py` so `SMPL_PCMPJPE` uses the translation-only `pcmpjpe_func` path.

## 2. Validation and docs

- [x] 2.1 Update `tests/test_pcmpjpe_metrics.py` for translation-only invariance and non-zero rotation mismatch cases.
- [x] 2.2 Update `docs/pcmpjpe_metrics.md` to describe translation-only semantics and remove orientation-alignment claims.
- [x] 2.3 Run `uv run pytest -q tests/test_pcmpjpe_metrics.py` and fix regressions.
