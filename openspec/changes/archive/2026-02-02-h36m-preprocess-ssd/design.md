## Context

H36M is currently read as full-resolution JPEG sequences from HDD, which is slow. The project already includes `tools/data_preprocess.py` routines for MMFi and Humman that downscale and re-encode inputs for SSD-friendly storage. We want an H36M preprocessing path that reduces RGB size (target 480×640) and stores compact metadata/poses for faster loading. This is a data pipeline change; compute-heavy model code stays the same and continues to run on GPU.

## Goals / Non-Goals

**Goals:**
- Add an H36M preprocessing script path in `tools/data_preprocess.py` that produces a compact SSD dataset (RGB resized to 480×640).
- Define a clear output folder layout and metadata for fast loading.
- Provide usage instructions aligned with existing preprocessing scripts.

**Non-Goals:**
- Changing model architectures or training recipes.
- Adding new modalities beyond RGB/3D joints.
- Building a new dataset format incompatible with existing H36M metadata.

## Decisions

- **Output resolution:** Resize RGB to 480×640 for SSD storage; JPEG encoding (quality ~95) to balance size vs fidelity.
- **Output layout:** Mirror existing preprocess outputs (per-modality subfolders) and include per-sequence metadata for fast indexing.
- **Pose storage:** Save GT 3D joints (and optionally derived 2D) in compact NumPy format (float16) alongside frames.
- **Loader strategy:** Add a preprocessed H36M loader or extend H36M dataset to read preprocessed paths via a flag, minimizing code duplication.
- **Compute placement:** Preprocessing CPU-only; training uses GPU as before.

## Risks / Trade-offs

- [Lossy JPEG artifacts] → Mitigation: use high quality (95) and document trade-off; allow configurable quality.
- [Increased preprocessing time] → Mitigation: multiprocessing and idempotent outputs.
- [Format drift vs original H36M] → Mitigation: keep metadata compatible with existing H36M indexing and document any differences.
