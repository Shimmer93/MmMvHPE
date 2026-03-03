## Why

H36M training is slow because the dataset is loaded from large image/video files on HDD. A preprocessing script that compacts data for SSD storage (as done for MMFi/Humman) will speed up loading and improve iteration time.

## What Changes

- Add an H36M preprocessing pipeline to `tools/data_preprocess.py` to generate a compact SSD-friendly format.
- Define the output layout and metadata needed for fast loading in H36M dataset classes.
- Provide usage instructions/flags consistent with existing MMFi/Humman preprocessing.

## Capabilities

### New Capabilities
- `h36m-preprocess-script`: H36M preprocessing routine that exports compact RGB/pose data for fast SSD loading.
- `h36m-preprocess-format`: Defined output format and metadata for the preprocessed H36M data.

### Modified Capabilities
- (none)

## Impact

- Affected code: `tools/data_preprocess.py`, datasets (H36M loader changes or new loader for preprocessed data), docs.
- Training/eval workflows: add a preprocessing step before training; dataset paths can point to SSD output.
- Modalities: RGB and 3D joints (depth optional depending on output choice).
