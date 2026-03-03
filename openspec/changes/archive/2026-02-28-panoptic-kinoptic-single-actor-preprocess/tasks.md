## 1. OpenSpec Finalization

- [x] 1.1 Create change folder and proposal for single-actor Panoptic Kinoptic preprocessing
- [x] 1.2 Create design covering synchronization, cropping, sequence-preserving output, and failure modes
- [x] 1.3 Add spec requirements for script behavior and output format

## 2. Implementation

- [x] 2.1 Add `tools/preprocess_panoptic_kinoptic.py` with CLI support for selected sequence preprocessing (`--sequences`, `--sequence-list`, `--max-sequences`)
- [x] 2.2 Implement strict sequence-level validation of required input artifacts (`ksynctables`, `kcalibration`, `hdPose3d_stage1_coco19`, Kinect RGB videos)
- [x] 2.3 Implement synchronization by `univTime` between `body3DScene` annotations and Kinect color/depth frames with configurable max time delta
- [x] 2.4 Implement HuMMan-style square RGB crop generation (YOLO-based) and compact resized output writing
- [x] 2.5 Preserve sequence-local output structure under `<out_root>/<sequence>/...` and write metadata (`sync_map`, crop params, manifest)
- [x] 2.6 Add clear per-sequence summary logging and `--continue-on-error` behavior

## 3. Validation

- [x] 3.1 Smoke-test on `/data/shared/panoptic-toolbox/161029_piano2` with a bounded frame count
- [x] 3.2 Verify output structure is sequence-preserving and synchronized frame count is non-zero
- [x] 3.3 Verify cropped RGB resolution and metadata presence

## 4. Documentation

- [x] 4.1 Update `docs/tools.md` with command examples for the new Panoptic preprocessing script
- [x] 4.2 Add a Panoptic preprocessing note in `docs/` describing assumptions, synchronization, and output layout
