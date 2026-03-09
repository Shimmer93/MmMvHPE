# SAM3D Panoptic Segment Evaluation

`tools/sam3d_panoptic_segment_eval/run_segment_eval.py` evaluates SAM-3D-Body on preprocessed Panoptic single-actor RGB data, scores fixed-length non-overlapping temporal segments, and writes logs and plots under `logs/sam3d_panoptic_segment_eval/`.

The analysis tool uses the Panoptic dataset class joint order, matching the preprocessed dataset and toolbox comments:

- `neck`
- `nose`
- `bodyCenter` / `mid_hip`
- left arm
- left leg
- right arm
- right leg
- left eye / ear
- right eye / ear

## Expected Config Shape

The entry script accepts a Panoptic config in the same style used by `scripts/visualize_sam3d_body_rerun.py`:

- `train_dataset` / `val_dataset` / `test_dataset`
- `train_pipeline` / `val_pipeline` / `test_pipeline`
- `vis_denorm_params`

The config should remain multiview. Camera selection is done with `--camera kinect_00X`, so one config can be reused across all RGB cameras.

Example config:

- [configs/analysis/panoptic_sam3d_segment_eval_office1.yml](/home/yzhanghe/MmMvHPE/configs/analysis/panoptic_sam3d_segment_eval_office1.yml)

## Segment Semantics

- One run accepts exactly one `--segment-length`
- Segments are contiguous and non-overlapping
- Segments never cross sequence boundaries
- Incomplete tails are dropped
- Metrics are aggregated per `(sequence_name, camera_name, segment_index)`

## Metrics

The tool reuses repository metric semantics from [metrics/mpjpe.py](/home/yzhanghe/MmMvHPE/metrics/mpjpe.py):

- `MPJPE`
- `PA-MPJPE`
- `PC-MPJPE`

Metrics are computed in meters in the selected RGB camera coordinate frame.

## Invalid Frames

`--invalid-frame-mode drop` keeps the run going and records invalid-frame bookkeeping in the segment logs.

Invalid frames include cases such as:

- SAM3D returns zero people
- SAM3D returns more than one person
- non-finite predicted or GT keypoints

`--invalid-frame-mode error` stops on the first invalid frame.

## Output Files

Each run directory contains:

- `segments.csv`
- `segments.json`
- `sequence_camera_summary.csv`
- `sequence_camera_summary.json`
- `worst_segments_by_<metric>.csv`
- `worst_segments_by_<metric>.json`
- `segment_metric_histograms.png`
- `worst_sequence_camera_pairs_<metric>.png`
- `run_summary.json`

If `--export-worst-k K` is used, the run also writes `worst_segment_exports/` with one directory per exported segment. Each exported segment contains:

- one overlay PNG per frame
- `segment_metadata.json`

Overlay PNGs show:

- GT Panoptic dataset joints projected to the selected RGB image in green
- SAM3D Panoptic dataset joints projection in orange

## SAM3 To Panoptic Joint Mapping

The adapter is implemented in [joint_adapter.py](/home/yzhanghe/MmMvHPE/tools/sam3d_panoptic_segment_eval/joint_adapter.py) and uses explicit named MHR70 joints from `sam_3d_body.metadata.mhr70`.

| Panoptic dataset joint | SAM3 MHR70 source |
|---|---|
| `neck` | `neck` |
| `nose` | `nose` |
| `mid_hip` | `0.5 * (left_hip + right_hip)` |
| `left_shoulder` | `left_shoulder` |
| `left_elbow` | `left_elbow` |
| `left_wrist` | `left_wrist` |
| `left_hip` | `left_hip` |
| `left_knee` | `left_knee` |
| `left_ankle` | `left_ankle` |
| `left_eye` | `left_eye` |
| `left_ear` | `left_ear` |
| `right_shoulder` | `right_shoulder` |
| `right_elbow` | `right_elbow` |
| `right_wrist` | `right_wrist` |
| `right_hip` | `right_hip` |
| `right_knee` | `right_knee` |
| `right_ankle` | `right_ankle` |
| `right_eye` | `right_eye` |
| `right_ear` | `right_ear` |

The adapter fails fast if one of these required source joints is missing from the SAM3 metadata.

## Example Commands

Score 8-frame segments on office1 / `kinect_008`:

```bash
uv run python tools/sam3d_panoptic_segment_eval/run_segment_eval.py \
  --cfg configs/analysis/panoptic_sam3d_segment_eval_office1.yml \
  --split test \
  --camera kinect_008 \
  --segment-length 8
```

Score and export the 5 worst segments by `MPJPE`:

```bash
uv run python tools/sam3d_panoptic_segment_eval/run_segment_eval.py \
  --cfg configs/analysis/panoptic_sam3d_segment_eval_office1.yml \
  --split test \
  --camera kinect_008 \
  --segment-length 8 \
  --export-worst-k 5 \
  --rank-metric mpjpe
```

## Multi-Sequence Worst-Case Sweep

For the office/cello comparison workflow, use:

- [run_panoptic_sam3d_worst_case_sweep.py](/home/yzhanghe/MmMvHPE/scripts/run_panoptic_sam3d_worst_case_sweep.py)

This script:

- evaluates all requested cameras across the office1 / office2 / cello3 analysis configs
- ranks the top-K worst sequence-camera pairs by each of `MPJPE`, `PA-MPJPE`, and `PC-MPJPE`
- ranks the top-K worst segments by each metric
- writes rerun spec files and `.rrd` overlay recordings for those selected worst items

Example:

```bash
uv run python scripts/run_panoptic_sam3d_worst_case_sweep.py \
  --segment-length 8 \
  --top-k 3 \
  --device cuda
```

Main outputs under `logs/panoptic_sam3d_worst_case_sweep/<run-name>/`:

- `all_pairs.csv`
- `all_segments.csv`
- `top_pairs_by_mpjpe.json`
- `top_pairs_by_pa_mpjpe.json`
- `top_pairs_by_pc_mpjpe.json`
- `top_segments_by_mpjpe.json`
- `top_segments_by_pa_mpjpe.json`
- `top_segments_by_pc_mpjpe.json`
- `rerun_specs/`
- `rerun/`
