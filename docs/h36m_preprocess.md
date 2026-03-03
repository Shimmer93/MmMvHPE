# H36M Preprocessing (SSD)

This script creates a compact H36M dataset under `/opt/data/h36m_preprocessed` with RGB resized to 480x640 and GT 3D joints stored as float16.

## Output Layout
```
/opt/data/h36m_preprocessed/
  rgb/    # JPEG frames, 480x640
  gt3d/   # per-frame 3D joints (.npy, float16)
  camera-parameters.json
  metadata.xml
```

## Run
```bash
uv run python tools/data_preprocess.py \
  --dataset h36m \
  --root_dir /data/shared/H36M-Toolbox \
  --out_dir /opt/data/h36m_preprocessed \
  --rgb_w 640 --rgb_h 480 \
  --num_workers 16
```

### Dry-run (subset)
```bash
uv run python tools/data_preprocess.py \
  --dataset h36m \
  --root_dir /data/shared/H36M-Toolbox \
  --out_dir /opt/data/h36m_preprocessed_dryrun \
  --rgb_w 640 --rgb_h 480 \
  --num_workers 1 \
  --max_sequences 2 \
  --max_frames 5
```

## Notes
- RGB is stored as JPEG (quality 95).
- Joints are stored in camera coordinates with 17 joints (static joints removed).
- TODO (recommended fix): scale camera intrinsics to match the resized RGB resolution (e.g., 640x480). The dataset loader currently scales intrinsics at load time as a temporary fix; prefer updating `camera-parameters.json` during preprocessing so intrinsics are stored in the correct pixel space.
