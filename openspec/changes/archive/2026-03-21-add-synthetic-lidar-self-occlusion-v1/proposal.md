## Why

The current synthetic LiDAR generation path only filters mesh samples by surface normal, so it keeps many points that would be hidden by self-occlusion from the virtual sensor viewpoint. That limits the realism of the synthetic point clouds and makes the saved dataset less useful for transfer experiments that depend on viewpoint-dependent visibility.

## What Changes

- Add a `v1` synthetic LiDAR regeneration workflow that reuses the saved synthetic mesh and virtual sensor pose to produce self-occlusion-aware point clouds.
- Add a depth-buffer visibility pass that removes self-occluded mesh samples before the final point-cloud export.
- Keep `v0-a` and `v1` LiDAR artifacts side by side in each synthetic sample and add versioned point-cloud metadata so the desired LiDAR version can be selected from dataset/config parameters.
- Add a dataset-scale regeneration script to update existing synthetic sample directories in place without rerunning mask generation, SAM-3D-Body, or GT export.
- Add optional QC rendering inside the regeneration workflow for comparing `v0-a` and `v1` point clouds during validation runs.

## Capabilities

### New Capabilities
- `synthetic-lidar-self-occlusion`: regenerate saved synthetic LiDAR point clouds with depth-buffer-based self-occlusion handling and explicit simulation-version metadata.

### Modified Capabilities
- `synthetic-exported-training-dataset`: select which saved synthetic LiDAR version to load from dataset/config parameters when multiple LiDAR artifacts coexist in a sample.

## Impact

- Affected code: `projects/synthetic_data/virtual_lidar.py`, synthetic pipeline/export scripts under `scripts/`, the synthetic exported dataset under `datasets/`, and synthetic-data docs under `docs/`.
- Affected data: existing synthetic sample folders under roots such as `/opt/data/coco/synthetic_data/v0a_train2017` will gain regenerated LiDAR artifacts and updated point-cloud metadata.
- Runtime impact: regeneration will be slower than the current normal-only sampler but will avoid rerunning the expensive RGB-to-mesh stages.
- Output impact: manifests and QC tooling in `logs/` will need to identify the LiDAR simulation mode/version used for each sample, and dataset configs will need a parameter for selecting the desired LiDAR artifact version.
