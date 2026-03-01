## 1. CLI and Frame Selection

- [x] 1.1 Add `--num-frames` argument to `scripts/visualize_sam3d_body_rerun.py` with default `1`.
- [x] 1.2 Implement deterministic frame-index selection utility that supports `frame-index >= 0` (start anchor) and `frame-index = -1` (center anchor), with bounds clipping.
- [x] 1.3 Keep existing single-frame behavior unchanged when `--num-frames` is omitted or `1`.

## 2. Multi-Frame Timeline Logging

- [x] 2.1 Refactor single-frame inference/logging flow in `scripts/visualize_sam3d_body_rerun.py` into a per-frame loop over selected frame indices.
- [x] 2.2 Log each visualized frame as a distinct rerun timeline step and add per-step metadata for source frame index under `world/info/*`.
- [x] 2.3 Ensure namespace conventions remain unchanged (`world/inputs/*`, `world/front/*`, `world/side/*`, `world/info/*`).

## 3. Render-Mode Compatibility

- [x] 3.1 Verify and enforce that `overlay` mode skips mesh logging for all frames in multi-frame mode.
- [x] 3.2 Verify and enforce that `auto` and `mesh` modes preserve existing semantics for every frame.

## 4. Validation and Documentation

- [x] 4.1 Run smoke validation with `--num-frames 1` to confirm backward compatibility.
- [x] 4.2 Run smoke validation with `--num-frames > 1` for at least `overlay` and `mesh` modes using `uv run` and save `.rrd` outputs under `logs/`.
- [x] 4.3 Update `docs/rerun_visualization.md` with `--num-frames` usage, frame-selection semantics, and examples.
