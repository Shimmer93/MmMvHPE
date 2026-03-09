## Context

MMHPE currently has two separate SAM-3D-Body workflows:
- config-driven rerun visualization in `scripts/visualize_sam3d_body_rerun.py`
- Panoptic dataset loading and split handling in `datasets/panoptic_preprocessed_dataset_v1.py`

Those workflows are sufficient for visual inspection, but they do not provide quantitative, segment-level failure analysis across sequence, camera view, and time. The requested change introduces a new analysis workflow that reuses the same config-driven dataset selection style as the current SAM3 visualization configs, but evaluates one selected RGB camera at a time and scores contiguous temporal segments rather than individual spot-check frames.

There are several constraints:
- The input config should stay multiview-friendly. Users should pass one Panoptic RGB dataset config and select the target camera through a CLI argument instead of maintaining one config per camera.
- The evaluation must work on `train`, `val`, or `test` splits already defined by the config and `split_config`.
- SAM-3D-Body output joints are not in Panoptic COCO19 format, so the workflow needs an explicit adapter before metric computation.
- Outputs must be easy to inspect after a long run: structured logs for sorting/filtering and plots that show where the worst segments are.
- The project already has MPJPE-family metric code and Panoptic joint conventions; this workflow should reuse those conventions rather than invent a parallel metric definition.

The implementation is cross-cutting but should remain incremental. It does not belong in `main.py` because it is not a training or standard evaluation loop. A dedicated analysis package under `tools/` is the cleanest fit.

## Goals / Non-Goals

**Goals:**
- Add a dedicated SAM3 Panoptic analysis directory under `tools/` that contains the evaluation entrypoint and helper modules.
- Accept a SAM3-style Panoptic config and a CLI camera selector such as `--camera kinect_008` so one config can cover all RGB views.
- Traverse one chosen split and partition the selected camera stream into fixed, non-overlapping temporal segments of configurable length such as `8`, `16`, or `32`.
- Run SAM-3D-Body inference frame-by-frame for the selected camera, convert prediction joints into Panoptic COCO19, and compute per-segment `MPJPE`, `PA-MPJPE`, and `PC-MPJPE`.
- Record per-segment metadata including sequence, camera, frame start/end, segment index, sample ids, and metric values.
- Write machine-readable logs and summary plots so the user can identify the worst-performing regions in the dataset.
- Add documentation under `docs/` with command examples and interpretation notes.

**Non-Goals:**
- Modifying the training pipeline, dataset preprocessing outputs, or `main.py`.
- Replacing the current rerun visualization scripts.
- Adding a generic benchmark framework for all datasets or all external models.
- Evaluating non-RGB modalities as SAM-3D-Body inputs.
- Supporting overlapping segments or complex sampling policies in the first version.

## Decisions

### 1. Create a dedicated analysis package under `tools/`

**Decision:** Add a new directory such as `tools/sam3d_panoptic_segment_eval/` with small focused modules:
- `run_segment_eval.py` or `main.py`: CLI entrypoint
- `dataset_adapter.py`: load config, select split, filter one camera, iterate samples
- `joint_adapter.py`: convert SAM3 output joints to Panoptic COCO19
- `segment_metrics.py`: aggregate frame metrics into per-segment metrics
- `reporting.py`: write CSV/JSON summaries and plots

**Why:** The work is analysis-specific and would make the rerun scripts harder to maintain if mixed in. A dedicated directory also satisfies the request to isolate this task.

**Alternative considered:** Extend `scripts/visualize_sam3d_body_rerun.py` with metric modes. Rejected because visualization and batch evaluation have different runtime/output contracts.

### 2. Reuse config-driven dataset loading, but make camera selection a CLI concern

**Decision:** The analysis entrypoint will accept:
- `--cfg <config>`
- `--split {train,val,test}`
- `--camera kinect_00X`
- `--segment-length {8,16,32,...}`

The config remains a normal Panoptic SAM config that may list many RGB cameras. The script will instantiate the dataset from that config, then override/filter to one camera after config load rather than requiring one config per camera.

**Why:** This keeps the evaluation aligned with the current config-driven workflow while avoiding a combinatorial config explosion across ten Kinect views.

**Alternative considered:** Require one config per camera. Rejected because it is high-maintenance and error-prone.

### 3. Segment by contiguous, non-overlapping sample windows

**Decision:** Build segments from the already materialized dataset order for the selected split and camera. For `seq_len: 1`, contiguous dataset samples correspond to synchronized body frames. Segments will be non-overlapping groups of `N` consecutive frames. Partial tail segments will be dropped by default in the first version.

Each segment record will include:
- `sequence_name`
- `camera_name`
- `segment_start_sample_idx`
- `segment_end_sample_idx`
- `frame_ids`
- `num_frames`

**Why:** Non-overlapping segments are simple, deterministic, and easier to rank. They also map directly to the user’s request for granularity buckets such as `8`, `16`, and `32`.

**Alternative considered:** Overlapping sliding windows. Rejected for the first version because it multiplies runtime and complicates worst-segment ranking.

### 4. Use a named-joint SAM3-to-Panoptic adapter rather than forcing SMPL24 conversion

**Decision:** The workflow will convert SAM-3D-Body prediction joints to Panoptic COCO19 through an explicit named-joint adapter derived from SAM3 metadata, not through a lossy “SMPL skeleton only” conversion.

The adapter will:
- inspect SAM3 joint names from the model metadata
- map direct semantic matches (nose, neck, shoulders, elbows, wrists, hips, knees, ankles, eyes, ears)
- derive `body_center` / `mid_hip` from left/right hip if needed
- fail fast if a required Panoptic joint cannot be mapped or derived unambiguously

**Why:** Panoptic COCO19 includes face joints and body-center semantics that are not faithfully represented by plain SMPL24. A named-joint adapter is more correct and easier to audit.

**Alternative considered:** Convert from SMPL24 only. Rejected because it cannot recover all Panoptic COCO19 joints cleanly and would weaken metric validity.

### 5. Reuse existing project metric implementations where possible

**Decision:** Frame-level metric computation will reuse the project’s MPJPE-family definitions and alignment conventions instead of reimplementing them ad hoc inside the script. The new workflow will compute metrics per frame, then aggregate by segment with mean, max, and count statistics.

Planned segment outputs:
- `segment_mpjpe_mean`
- `segment_pa_mpjpe_mean`
- `segment_pc_mpjpe_mean`
- optional per-frame arrays in JSON for deep inspection

**Why:** This keeps Panoptic/SAM3 analysis consistent with existing MMHPE evaluation semantics.

**Alternative considered:** Reimplement all three metrics locally in the script. Rejected because it would create drift from the rest of the repository.

### 6. Write both sortable logs and plots

**Decision:** Each run will write a dedicated output directory under `logs/`, for example:
- `logs/sam3d_panoptic_segment_eval/<run_name>/segments.csv`
- `logs/sam3d_panoptic_segment_eval/<run_name>/segments.json`
- `logs/sam3d_panoptic_segment_eval/<run_name>/worst_segments.csv`
- `logs/sam3d_panoptic_segment_eval/<run_name>/metric_histograms.png`
- `logs/sam3d_panoptic_segment_eval/<run_name>/sequence_camera_heatmap.png`

The run name will encode config, split, camera, and segment length.

**Why:** CSV/JSON serve sorting and downstream scripting; plots give a quick visual answer to “where does SAM perform worst?”

**Alternative considered:** Only plots or only console output. Rejected because the user explicitly wants both a log file and a graph.

### 7. Keep runtime simple and explicit

**Decision:** The first version will run inference in a single-process script over one selected camera stream. It will not integrate with distributed Lightning evaluation or reuse `main.py`.

**Why:** This workflow is analysis-heavy and model-external. Simplicity is more valuable than framework integration here.

**Alternative considered:** Add a new evaluation mode to `main.py`. Rejected because it would add complexity to the main training/evaluation pipeline for a niche external-model analysis task.

## Risks / Trade-offs

- [Risk] Joint mapping from SAM3 output to Panoptic COCO19 is wrong or incomplete. → Mitigation: build the adapter from named SAM3 metadata, document the mapping table, and fail fast on missing joints instead of silently approximating unsupported semantics.
- [Risk] The chosen config may apply transforms that invalidate GT for metric evaluation. → Mitigation: require evaluation configs that preserve metric-space GT semantics, or explicitly undo transforms when reading the sample; document acceptable config assumptions.
- [Risk] Segment ranking becomes misleading if segments cross sequence boundaries. → Mitigation: segment only within one `(sequence, camera)` stream and never merge frames from different sequences into the same segment.
- [Risk] Full-dataset SAM3 inference is slow. → Mitigation: keep the camera argument explicit, use non-overlapping segments, cache per-frame predictions inside one run, and log progress/ETA.
- [Risk] Plot generation becomes a maintenance burden. → Mitigation: limit first-version plots to static PNG summaries using existing plotting dependencies already available in the environment.
- [Risk] Backward compatibility confusion with current SAM3 rerun configs. → Mitigation: keep the analysis entrypoint separate and document that rerun configs are accepted as dataset selectors, not as direct visualization configs.

## Migration Plan

1. Add the new analysis directory under `tools/` and implement the segment-evaluation entrypoint plus helpers.
2. Add one or more Panoptic SAM analysis configs or examples in `configs/vis/` or a dedicated config subdirectory.
3. Add docs under `docs/` with:
   - required environment/checkpoint assumptions
   - command examples
   - explanation of segment logs and plots
4. Validate on a small subset first (single sequence, single camera, short segment length).
5. Run a full-sequence evaluation and inspect the generated CSV/plots for worst-segment reporting.

Rollback is simple: remove or ignore the new analysis directory and configs. No existing training or visualization entrypoints need to change.

## Open Questions

- None for the first implementation. The current design assumes dropped tail segments, one segment length per run, sequence-camera aggregation, and optional top-K worst-segment rerun export support.
