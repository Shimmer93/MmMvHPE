## 1. Unify Visualization Sampling Semantics

- [x] 1.1 Extract shared frame-selection helper(s) under `scripts/rerun_utils/` and use them in both `scripts/visualize_inference_rerun.py` and `scripts/visualize_sam3d_body_rerun.py`.
- [x] 1.2 Implement explicit `seq_len=1` cross-sample stepping behavior for `--num-frames` in `scripts/visualize_inference_rerun.py` while preserving config-driven model input construction.
- [x] 1.3 Ensure `sample-idx`, `frame-index`, and `num-frames` behavior is consistent between both scripts for equivalent sample/window/frame selections.

## 2. Temporal GT Consumption

- [x] 2.1 Keep optional temporal SMPL GT output in `datasets/humman_dataset_v2.py` and pass-through support in `datasets/humman_dataset_v3.py` via `return_smpl_sequence`.
- [x] 2.2 Update `scripts/visualize_inference_rerun.py` to prefer `gt_keypoints_seq`/`gt_smpl_params_seq` and fallback to single-frame GT fields.
- [x] 2.3 Update `scripts/visualize_sam3d_body_rerun.py` to prefer `gt_smpl_params_seq` and fallback to `gt_smpl_params`.

## 3. Demo Config Alignment

- [x] 3.1 Set demo configs to current model-compatible `seq_len: 1` in `configs/demo/humman_smpl_v5_kcam_gcn_mix_synth_v3.yml` and `configs/demo/humman_sam3d_body_vis.yml`.
- [x] 3.2 Keep temporal GT flags enabled in demo configs (`return_keypoints_sequence`, `return_smpl_sequence`) for visualization compatibility.

## 4. Validation

- [x] 4.1 Run syntax checks: `uv run --no-sync python -m py_compile scripts/visualize_inference_rerun.py scripts/visualize_sam3d_body_rerun.py`.
- [x] 4.2 Run inference rerun smoke test with multi-frame request and verify `.rrd` output: `uv run --no-sync python scripts/visualize_inference_rerun.py -c configs/demo/humman_smpl_v5_kcam_gcn_mix_synth_v3.yml --checkpoint <ckpt> --split train --sample-idx 0 --num-frames 50 --coord-space sensor --reference-sensor lidar --reference-view 0 --no_serve --save_rrd logs/rerun_vis/mmmvhpe_train_ref_lidar_50.rrd`.
- [x] 4.3 Run SAM rerun smoke test and verify `.rrd` output: `uv run --no-sync python scripts/visualize_sam3d_body_rerun.py -c configs/demo/humman_sam3d_body_vis.yml --split train --sample-idx 0 --num-frames 50 --render-mode mesh --gt-coordinate-space camera --no_serve --save_rrd logs/rerun_vis/sam3d_train_50.rrd`.

## 5. Documentation

- [x] 5.1 Add/update docs under `docs/` describing config-driven model input (`seq_len`) vs CLI-driven visualization sampling.
- [x] 5.2 Add concrete command examples for both scripts, including `seq_len=1` behavior and temporal GT fallback behavior.
