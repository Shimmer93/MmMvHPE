## Why

SAM-3D-Body rerun demos are useful for spot checks, but they do not tell us where the model fails across Panoptic sequences, views, and temporal regions. We need a reproducible analysis workflow that scores SAM-3D-Body on config-selected Panoptic splits, aggregates errors over fixed temporal segments, and surfaces the worst-performing sequence/view/frame regions with logs and plots.

## What Changes

- Add a config-driven SAM-3D-Body segment evaluation workflow for Panoptic preprocessed datasets that accepts the same style of dataset config currently used for SAM3 visualization.
- Add a dedicated analysis package/directory for this task so segment evaluation code, helpers, configs, and outputs are isolated from rerun-only visualization scripts.
- Add dataset traversal logic that groups one selected split (`train`, `val`, or `test`) into fixed-length segments with configurable granularity such as `8`, `16`, or `32` frames.
- Add metric computation per segment for `MPJPE`, `PA-MPJPE`, and `PC-MPJPE`, with explicit joint-format conversion from SAM-3D-Body output skeletons to Panoptic COCO19 ground truth.
- Add structured result logging that records per-segment metadata such as sequence name, sensor/view, frame range, sample ids, and metric values.
- Add plot generation so each run writes graphs and sortable summaries that make the worst-performing segments easy to identify.
- Add command-line entrypoints and usage docs for running the analysis on different SAM3 configs, splits, and segment sizes.

## Capabilities

### New Capabilities
- `sam3d-panoptic-segment-evaluation`: Evaluate SAM-3D-Body predictions against Panoptic ground truth over fixed temporal segments, convert predictions into Panoptic COCO19 joints, compute segment metrics, and write logs/plots for failure analysis.

### Modified Capabilities
- None.

## Impact

- Affected code: new analysis scripts/modules under a dedicated directory, Panoptic/SAM3 evaluation helpers, joint conversion utilities, metric aggregation/report generation, and documentation/examples.
- Affected datasets/modalities: Panoptic preprocessed dataset, primarily RGB-driven SAM-3D-Body inference with Panoptic COCO19 GT; LiDAR/depth/mmWave are not inference inputs here unless only used for metadata or filtering.
- Runtime outputs: new logs and plots under `logs/` for segment-level evaluation summaries and worst-segment reporting.
- Dependencies/APIs: builds on existing SAM-3D-Body environment and Panoptic config loading; no intended breaking change to current training or rerun visualization entrypoints.
