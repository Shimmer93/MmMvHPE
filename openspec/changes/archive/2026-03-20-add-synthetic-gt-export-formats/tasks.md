## 1. Export Contract Design

- [x] 1.1 Document the training-facing target contracts for HuMMan and Panoptic in a new synthetic-export design note under `docs/`.
- [x] 1.2 Define explicit topology and coordinate-space metadata for exported synthetic GT bundles under `projects/synthetic_data/`.
- [x] 1.3 Define the target output layout and manifest schema for minimal format-specific exports on top of existing synthetic sample directories, without dataset-tree replication.

## 2. MHR To SMPL Integration

- [x] 2.1 Add an adapter module under `projects/synthetic_data/` for the upstream MHR-to-SMPL conversion workflow referenced in the MHR README.
- [x] 2.2 Add environment and dependency checks for the conversion path, including required SMPL model files.
- [x] 2.3 Save conversion status, selected backend, and fitting-quality metadata in the export manifest.

## 3. Target-Format Exporters

- [x] 3.1 Implement a HuMMan-oriented exporter that saves SMPL24 `gt_keypoints`, `gt_smpl_params`, camera metadata, LiDAR input, and LiDAR-centered GT variants.
- [x] 3.2 Implement a Panoptic-oriented exporter that saves Panoptic COCO19 `gt_keypoints` with explicit topology metadata and no synthetic SMPL parameter payload.
- [x] 3.3 Add explicit coordinate-space variants and manifest entries for canonical/new-world, LiDAR, and PC-centered LiDAR outputs.

## 4. Validation

- [x] 4.1 Add a smoke-test script or CLI mode that runs the exporter on already generated synthetic samples from COCO val.
- [x] 4.2 Validate exported camera metadata with the current `CameraParamToPoseEncoding` path using `uv run`.
- [x] 4.3 Validate LiDAR-centered exports against the current `PCCenterWithKeypoints` contract using `uv run`.

## 5. Documentation

- [x] 5.1 Update `docs/sam3d_synthetic_data_v0a.md` with the new export stage and example commands.
- [x] 5.2 Add a dedicated exporter usage document under `docs/` with format descriptions, required dependencies, and output examples.
