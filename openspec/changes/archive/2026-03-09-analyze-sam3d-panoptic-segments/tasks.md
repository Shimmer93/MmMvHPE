## 1. Analysis Package Setup

- [x] 1.1 Create a dedicated analysis directory under `tools/sam3d_panoptic_segment_eval/` with the entry script and helper modules described in the design (`run_segment_eval.py`, `dataset_adapter.py`, `joint_adapter.py`, `segment_metrics.py`, `reporting.py`).
- [x] 1.2 Add a small shared run-output helper under `tools/sam3d_panoptic_segment_eval/` that creates deterministic run directories under `logs/sam3d_panoptic_segment_eval/`.
- [x] 1.3 Add at least one reusable Panoptic SAM analysis config example under `configs/` that keeps multiview RGB cameras in config and relies on CLI `--camera` selection instead of per-camera config duplication.

## 2. Dataset Traversal And Segment Building

- [x] 2.1 Implement config loading and dataset creation in `tools/sam3d_panoptic_segment_eval/dataset_adapter.py` using the same merge path currently used by `scripts/visualize_sam3d_body_rerun.py`.
- [x] 2.2 Implement single-camera filtering in `tools/sam3d_panoptic_segment_eval/dataset_adapter.py` so one multiview config can be evaluated with `--camera kinect_00X`.
- [x] 2.3 Implement deterministic non-overlapping segment construction in `tools/sam3d_panoptic_segment_eval/dataset_adapter.py` for one selected split and one selected camera.
- [x] 2.4 Implement tail dropping in `tools/sam3d_panoptic_segment_eval/dataset_adapter.py` so incomplete final segments are excluded from scoring.
- [x] 2.5 Add validation for missing camera / empty split / cross-sequence segment boundary violations with explicit errors.

## 3. SAM3 Joint Adaptation And Metrics

- [x] 3.1 Implement a named-joint SAM3-to-Panoptic COCO19 adapter in `tools/sam3d_panoptic_segment_eval/joint_adapter.py`, including explicit mapping for direct semantic matches and derived `mid_hip`.
- [x] 3.2 Document the final joint mapping table in `docs/` and make the adapter fail fast if a required Panoptic joint cannot be mapped or derived.
- [x] 3.3 Implement frame-level metric evaluation in `tools/sam3d_panoptic_segment_eval/segment_metrics.py` by reusing the repository’s existing `MPJPE`, `PA-MPJPE`, and `PC-MPJPE` semantics.
- [x] 3.4 Implement per-segment aggregation in `tools/sam3d_panoptic_segment_eval/segment_metrics.py` with at least mean, max, and valid-frame counts for all three metrics.
- [x] 3.5 Add explicit handling for invalid frames in `tools/sam3d_panoptic_segment_eval/segment_metrics.py` so bookkeeping is visible and no frame is silently treated as zero error.

## 4. Reporting And Worst-Segment Export

- [x] 4.1 Implement structured result writing in `tools/sam3d_panoptic_segment_eval/reporting.py` for per-segment CSV and JSON logs.
- [x] 4.2 Implement grouped aggregation by `(sequence_name, camera_name)` in `tools/sam3d_panoptic_segment_eval/reporting.py`.
- [x] 4.3 Implement ranked worst-segment summaries and static plots in `tools/sam3d_panoptic_segment_eval/reporting.py`.
- [x] 4.4 Implement optional top-K worst-segment visualization export in `tools/sam3d_panoptic_segment_eval/run_segment_eval.py`, including metric-based ranking selection and output metadata.
- [x] 4.5 Ensure the run output directory contains enough metadata to trace every exported worst segment back to sequence, camera, frame ids, and ranking metric.

## 5. CLI, Validation, And Docs

- [x] 5.1 Implement the CLI in `tools/sam3d_panoptic_segment_eval/run_segment_eval.py` with `--cfg`, `--split`, `--camera`, `--segment-length`, and optional worst-segment export arguments.
- [x] 5.2 Add progress logging and runtime summaries in `tools/sam3d_panoptic_segment_eval/run_segment_eval.py` so long SAM3 runs remain monitorable.
- [x] 5.3 Add usage and behavior documentation in `docs/` covering config expectations, camera selection, segment semantics, output files, and worst-segment export.
- [x] 5.4 Validate the entrypoint on a small Panoptic subset with a runnable command such as `uv run python tools/sam3d_panoptic_segment_eval/run_segment_eval.py --cfg <config> --split test --camera kinect_008 --segment-length 8`.
- [x] 5.5 Validate worst-segment export with a runnable command such as `uv run python tools/sam3d_panoptic_segment_eval/run_segment_eval.py --cfg <config> --split test --camera kinect_008 --segment-length 8 --export-worst-k 5 --rank-metric mpjpe`.
