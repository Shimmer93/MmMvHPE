## Context

MMHPE already has SSD-friendly preprocessing routines in `tools/data_preprocess.py` for MMFi, HuMMan, and H36M. For Panoptic Kinoptic single-actor usage, we need a dedicated preprocessing flow that:
- synchronizes modalities using released Kinoptic timing metadata,
- produces HuMMan-cropped-style RGB outputs for compact/faster loading,
- preserves strict per-sequence boundaries,
- supports partial preprocessing for selected sequences while downloads are still ongoing.

The expected raw source layout is sequence-local under a Panoptic toolbox root, e.g.:
- `<root>/<seq>/kinectVideos/kinect_50_XX.mp4`
- `<root>/<seq>/kinect_shared_depth/KINECTNODEX/depthdata.dat`
- `<root>/<seq>/ksynctables_<seq>.json`
- `<root>/<seq>/kcalibration_<seq>.json`
- `<root>/<seq>/hdPose3d_stage1_coco19/body3DScene_*.json`

## Goals / Non-Goals

**Goals:**
- Add a Panoptic Kinoptic single-actor preprocessing script under `tools/`.
- Synchronize RGB/depth/body annotations per frame using sequence-local time metadata.
- Generate cropped RGB (HuMMan-cropped style: square crop + resize) for compact storage.
- Keep output grouped by sequence (`<out_root>/<seq>/...`) with no cross-sequence mixing.
- Allow explicit sequence selection via CLI (`--sequences` and/or `--sequence-list`).
- Fail fast on missing critical metadata per selected sequence, with explicit errors.

**Non-Goals:**
- Multi-actor support.
- Broad panoptic legacy (`hdPose3d_stage1`) compatibility in this change.
- Changes to model architecture or dataloader contract beyond preprocessing artifacts.
- Full benchmark/performance tuning beyond correctness and practical throughput.

## Decisions

- **Dedicated script instead of extending current monolithic preprocess CLI.**
  Implement in a new tool file (e.g. `tools/preprocess_panoptic_kinoptic.py`) to keep Panoptic-specific assumptions isolated and avoid adding brittle branches to existing MMFi/HuMMan/H36M paths.

- **Synchronization by universal time (`univTime`) rather than index alignment.**
  `body3DScene_*.json` frame IDs and Kinect video/depth frame indices are not directly aligned. Use `ksynctables_<seq>.json` (`kinect.color/depth.*.univ_time`) to map each annotation timestamp to nearest sensor frames with a strict max-time-delta threshold.

- **Crop source from RGB detections, applied per camera consistently.**
  Use a person bbox detector (same YOLO dependency pattern used by HuMMan-cropped preprocessing) to compute square crop windows. Reuse crop parameters for all synchronized frames of a `(sequence, camera)` stream to avoid temporal jitter and reduce preprocessing overhead.

- **Preserve sequence-local output tree.**
  Output format remains sequence-first:
  - `<out_root>/<seq>/rgb/<camera>/<frame>.jpg`
  - `<out_root>/<seq>/depth/<camera>/<frame>.png` (if depth export enabled)
  - `<out_root>/<seq>/meta/sync_map.json`
  - `<out_root>/<seq>/meta/crop_params.json`
  - optional compact GT copy under `<out_root>/<seq>/gt3d/`.

- **Strict input validation and explicit failure.**
  For each selected sequence, require presence/readability of `ksynctables`, `kcalibration`, `hdPose3d_stage1_coco19`, and at least one Kinect RGB stream. Missing critical files should fail the sequence with explicit message; `--continue-on-error` controls whether to proceed with other sequences.

- **Sequence selection controls are mandatory for operational flexibility.**
  Support:
  - `--sequences seq1,seq2,...`
  - `--sequence-list <txt>` (one sequence per line)
  - `--max-sequences` (for smoke tests)

## Data Flow

1. Resolve selected sequence set from CLI.
2. Validate required files for each sequence.
3. Build per-sequence synchronized frame pairs/triples:
   - read annotation `univTime` from each `body3DScene_*.json`.
   - map to nearest Kinect color/depth frame indices per node using `ksynctables`.
   - keep only matches within `max_sync_delta_ms`.
4. Compute/lookup crop windows per camera (YOLO over sampled frames).
5. Decode source frames, crop+resize, write compact outputs under sequence folder.
6. Write sequence-local metadata (`sync_map`, crop params, preprocessing manifest).

## Risks / Trade-offs

- **Risk: inaccurate or unstable crops if detector misses.**
  Mitigation: use sampled-frame aggregation per camera and fallback to full-frame crop if detector fails all samples.

- **Risk: synchronization drops many frames with strict threshold.**
  Mitigation: expose `--max-sync-delta-ms` and report per-sequence kept/dropped counts.

- **Risk: incomplete downloads during preprocessing.**
  Mitigation: sequence allowlist + `--continue-on-error` + per-sequence fail-fast validation.

- **Trade-off: preprocessing time vs runtime speed.**
  Preprocessing incurs one-time decode/crop cost but significantly reduces repeated training-time decode and random-I/O overhead.

## Validation Plan

Primary smoke validation target:
- `161029_piano2` under `/data/shared/panoptic-toolbox/161029_piano2`.

Validation checks:
- script runs on selected sequence only,
- outputs written under `<out_root>/161029_piano2/...` with sequence structure intact,
- synchronized mapping exists and frame count > 0,
- cropped RGB outputs are generated at configured output size,
- sequence summary reports kept/dropped/failed counts clearly.

## Documentation Updates

- Add script usage and examples to `docs/tools.md`.
- Add a focused Panoptic preprocessing note (new doc file) describing:
  - required input artifacts,
  - synchronization rule,
  - output structure,
  - known constraints (single-actor coco19 only for this change).
