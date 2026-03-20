# SAM3D GT Consistency Check

This script runs SAM-3D-Body on selected RGB frames from HuMMan and Panoptic, converts predictions into the target joint format used by each dataset, compares them against the dataset ground truth, and saves visual QC panels.

Current target formats:

- HuMMan: official MHR-to-SMPL conversion, compared as `SMPL24` in camera coordinates
- Panoptic: direct `MHR70 -> Panoptic19` adaptation, compared as `Panoptic19` in camera coordinates

The script writes:

- per-sample PNG figures under `logs/sam3d_gt_consistency_check/<run-name>/<dataset>/figures/`
- one contact sheet per dataset at `.../<dataset>/index.png`
- metric summary JSON at `.../summary.json`

Example:

```bash
uv run python scripts/run_sam3d_gt_consistency_check.py \
  --checkpoint-root /opt/data/SAM_3dbody_checkpoints \
  --humman-root /opt/data/humman_cropped \
  --panoptic-root /opt/data/panoptic_kinoptic_single_actor_cropped \
  --panoptic-sequences 171026_cello3 \
  --humman-max-samples 4 \
  --panoptic-max-samples 4 \
  --run-name smoke
```

Notes:

- HuMMan samples come from the existing `cross_camera_split` test split.
- Panoptic samples are loaded directly from the sequence(s) passed in `--panoptic-sequences`; they do not need to belong to the repo's predefined split config.
- The 3D plots are shown in camera coordinates because that is the frame used for the metric comparison.
