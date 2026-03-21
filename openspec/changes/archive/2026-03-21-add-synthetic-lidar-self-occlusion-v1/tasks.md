## 1. Point-Cloud Generator

- [x] 1.1 Add a `v1` self-occlusion-aware LiDAR generation path in `projects/synthetic_data/virtual_lidar.py`.
- [x] 1.2 Implement depth-buffer visibility filtering in the virtual LiDAR sensor frame with explicit configuration parameters.
- [x] 1.3 Extend synthetic sample metadata to record LiDAR simulation version, visibility parameters, and point counts before/after filtering while preserving the existing `v0-a` LiDAR artifact.

## 2. Regeneration Scripts

- [x] 2.1 Add a per-sample regeneration CLI for updating one synthetic sample directory from saved mesh and sensor artifacts.
- [x] 2.2 Add a dataset-scale resumable regeneration CLI for existing synthetic roots such as `/opt/data/coco/synthetic_data/v0a_train2017`.
- [x] 2.3 Implement the sibling-artifact strategy for `v1` outputs so `v0-a` and `v1` LiDAR artifacts coexist in each sample.
- [x] 2.4 Add optional QC rendering to the regeneration workflow and a flag to disable it for bulk processing.

## 3. Dataset Integration

- [x] 3.1 Update the synthetic exported dataset loader so LiDAR artifact version is selected from dataset/config parameters.
- [x] 3.2 Validate that the selected LiDAR version propagates cleanly through the existing synthetic pretrain configs without breaking batch collation.

## 4. Validation And QC

- [x] 4.1 Add QC support that compares `v0-a` and `v1` point clouds side by side for selected samples.
- [x] 4.2 Benchmark `512x512`, `720x720`, and `1024x1024` rendered depth-map resolutions on a validation subset and estimate full-dataset runtime for each.
- [x] 4.3 Compare the corresponding point-cloud quality across those resolution settings and choose the default depth-buffer resolution.
- [x] 4.4 Record the regeneration workflow, benchmarking procedure, and chosen default in `docs/`.
