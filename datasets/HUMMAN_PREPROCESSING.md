# Humman Dataset Preprocessing

## Generate 3D Keypoints from SMPL

Before using the Humman dataset, you need to precompute 3D keypoints from SMPL parameters.

### Prerequisites

1. **Download SMPL models** from [SMPL website](https://smpl.is.tue.mpg.de/)
2. Extract the models to a directory (e.g., `models/smpl/native/models`)
3. The directory should contain: `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`

### Usage

Run the preprocessing script:

```bash
conda activate mmhpe

python tools/generate_humman_keypoints.py \
    --data_root /data/shared/humman_release_v1.0_point \
    --smpl_model_path models/smpl/native/models \
    --gender neutral \
    --device cuda
```

### Options

- `--data_root`: Path to Humman dataset (default: `/data/shared/humman_release_v1.0_point`)
- `--smpl_model_path`: Path to SMPL model files (default: `models/smpl/native/models`)
- `--gender`: SMPL model gender - `neutral`, `male`, or `female` (default: `neutral`)
- `--device`: Device for computation - `cuda` or `cpu` (default: `cuda`)
- `--force`: Force regeneration even if keypoints already exist

### Output

The script will create `keypoints_3d.npz` in each sequence directory with:
- `keypoints_3d`: (N, 24, 3) array of 3D SMPL joint positions
- `num_frames`: Number of frames in the sequence

### Example

```bash
# Process all sequences
python tools/generate_humman_keypoints.py

# Force regenerate all keypoints
python tools/generate_humman_keypoints.py --force

# Use CPU instead of GPU
python tools/generate_humman_keypoints.py --device cpu
```

### SMPL Joint Format

The generated keypoints contain 24 SMPL joints:
- 0: Pelvis
- 1: Left Hip
- 2: Right Hip
- 3: Spine1
- 4: Left Knee
- 5: Right Knee
- 6: Spine2
- 7: Left Ankle
- 8: Right Ankle
- 9: Spine3
- 10: Left Foot
- 11: Right Foot
- 12: Neck
- 13: Left Collar
- 14: Right Collar
- 15: Head
- 16: Left Shoulder
- 17: Right Shoulder
- 18: Left Elbow
- 19: Right Elbow
- 20: Left Wrist
- 21: Right Wrist
- 22: Left Hand
- 23: Right Hand

You can convert to other skeleton formats (e.g., SimpleCOCO with 13 joints) using functions in `misc/skeleton.py`:
```python
from misc.skeleton import smpl2simplecoco
simplified_keypoints = smpl2simplecoco(keypoints_3d)  # (N, 13, 3)
```
