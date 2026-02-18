# Fixed-Sensor Evaluation Algorithm (Paper Summary)

This document summarizes the algorithm implemented in `tools/eval_fixed_lidar_frame.py` for evaluating canonical 3D skeletons in a fixed sensor coordinate frame.

## 1. Goal

Given predicted canonical 3D joints and ground-truth canonical 3D joints, evaluate error after projection into a physically meaningful, fixed sensor frame (typically LiDAR).

The script reports:
- MPJPE
- PA-MPJPE
- pelvis-centered MPJPE

## 2. Inputs and Notation

For sample `i`:
- Predicted canonical joints: `P_i in R^(J x 3)`
- Ground-truth canonical joints: `G_i in R^(J x 3)`
- Sample ID with sequence identity: `s_i` (e.g., `pXXXXXX_aXXXXXX_...`)

For modality `m`:
- Predicted camera pose encoding: `c_pred(i,m)`
- Ground-truth camera pose encoding: `c_gt(i,m)`
- Pose encoding type: `absT_quaR_FoV = [T_x,T_y,T_z,q_x,q_y,q_z,q_w,fov_h,fov_w]`

Decoded extrinsics are `E = [R|t] in R^(3x4)` with camera transform:

`x_cam = R x_world + t`

## 3. Camera Decoding

For `absT_quaR_FoV`, the script directly:
1. normalizes quaternion `(q_x,q_y,q_z,q_w)`,
2. converts quaternion to rotation matrix `R`,
3. forms `E = [R|t]`.

This avoids unnecessary Torch overhead and is numerically stable for batch-size-1 decoding during evaluation.

## 4. Sequence-Fixed Sensor Assumption

Sensors are assumed fixed per sequence. Therefore, for each sequence `s` and modality `m`, a single reference camera is chosen:
- the first finite camera encoding found in that sequence.

This is done independently for prediction and GT:
- `E_pred(s,m)`
- `E_gt(s,m)`

If a modality is absent or invalid for a sequence, that `(s,m)` reference is unavailable.

## 5. Projection Modes

## 5.1 `seq_lidar_ref` (baseline)

For each sample in sequence `s`:
- `P_i' = T(P_i ; E_pred(s,lidar))`
- `G_i' = T(G_i ; E_gt(s,lidar))`

where `T(X;[R|t]) = X R^T + t`.

This is the original single-LiDAR reference method.

## 5.2 `multi_sensor` (sequence-fixed multi-modal fusion)

Choose:
- target modality `tgt` (default `lidar`)
- fusion set `M_fuse` (default `{rgb,lidar}`)

For each sample in sequence `s` and modality `m in M_fuse` with valid refs:

1. Build homogeneous transforms:
- `H_pred(s,m)` from `E_pred(s,m)`
- `H_gt(s,m)` from `E_gt(s,m)`
- `H_gt(s,tgt)` from `E_gt(s,tgt)`

2. Map modality `m` to target frame using GT cross-sensor geometry:

`H_(m->tgt)(s) = H_gt(s,tgt) * inv(H_gt(s,m))`

3. Transport predicted camera to target frame:

`H_pred^tgt(s,m) = H_(m->tgt)(s) * H_pred(s,m)`

4. Project predicted joints through this target-frame camera:

`P_i^(m) = T(P_i ; E_pred^tgt(s,m))`

GT projection is done once in target frame:

`G_i' = T(G_i ; E_gt(s,tgt))`

Then fuse `{P_i^(m)}` according to Section 6 to obtain `P_i'`.

## 6. Reliability and Robust Fusion

The robust pipeline has two parts:
- reliability scoring per modality
- fusion policy

## 6.1 Reliability sources

### A) `cross_sensor`

Target-anchored disagreement score:
- target modality score is fixed to `1.0`
- for `m != tgt`:

`e_m = mean_j || P_i^(m,j) - P_i^(tgt,j) ||_2`

`r_cross(m) = exp(-e_m / tau_cross)`

where `tau_cross = --cross-sensor-tau` (meters).

If target is unavailable (rare), fallback is consensus-median disagreement.

### B) `temporal`

Computed once per sequence and modality from predicted cameras in that sequence:
- translation stability: std of `t` (mean across xyz axes)
- rotation stability: std of angular distance to first sequence rotation

Score:

`r_temp = exp(-(sigma_t / tau_t + sigma_r / tau_r))`

with:
- `tau_t = --temporal-trans-tau` (meters)
- `tau_r = --temporal-rot-tau-deg` (degrees)

### C) `hybrid`

`r(m) = r_cross(m) * r_temp(m)`

## 6.2 Fusion modes

Given reliability `r(m)`:

### `mean`
- uniform average over active modalities.

### `weighted`
- normalize reliability weights:

`w_m = r(m) / sum_k r(k)`

- fused prediction:

`P_i' = sum_m w_m * P_i^(m)`

### `hard_gate`
- keep modalities satisfying:

`r(m) >= gamma * max_k r(k)` with `gamma = --hard-gate-ratio`

- if none pass, keep only argmax modality
- fuse kept modalities with weighted rule above

This mode can automatically collapse to LiDAR-only when RGB camera reliability is low.

## 7. Optional Temporal Smoothing

After projection/fusion and before metrics, optional sequence-wise temporal smoothing is applied on predicted and/or GT joints:
- odd window size `--smooth-window`
- target chosen by `--smooth-on {pred,gt,both}`

## 8. Sample Filtering

A sample is dropped if any required condition fails, including:
- invalid/missing keypoints
- shape mismatch
- non-finite values
- unavailable required sequence references
- no valid modality remaining after projection

## 9. Complexity

Let:
- `N` = number of samples
- `J` = number of joints
- `M` = number of active fusion modalities

Runtime is approximately:
- sequence-ref build: `O(N * M)`
- projection/fusion: `O(N * M * J)`

Memory overhead is linear in number of sequences and modalities (reference cameras + optional temporal reliability stats).

## 10. Recommended Reporting in Papers

To ensure reproducibility, report:
- projection mode (`seq_lidar_ref` or `multi_sensor`)
- target modality
- fusion modalities
- fusion mode
- reliability source
- reliability hyperparameters (`tau_cross`, `tau_t`, `tau_r`, `hard_gate_ratio`)
- smoothing settings

Suggested ablation:
1. `seq_lidar_ref`
2. `multi_sensor + mean`
3. `multi_sensor + weighted + cross_sensor`
4. `multi_sensor + hard_gate + hybrid`

## 11. Reproducible Commands

Baseline:

```bash
uv run python tools/eval_fixed_lidar_frame.py \
  --pred-file <predictions.pkl> \
  --projection-mode seq_lidar_ref
```

Robust multi-sensor (recommended when RGB camera quality is unstable):

```bash
uv run python tools/eval_fixed_lidar_frame.py \
  --pred-file <predictions.pkl> \
  --projection-mode multi_sensor \
  --target-modality lidar \
  --fusion-modalities rgb,lidar \
  --fusion-mode hard_gate \
  --reliability-source hybrid \
  --hard-gate-ratio 0.8
```
