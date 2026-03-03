## 1. Preprocess Script

- [x] 1.1 Add H36M preprocessing mode in `tools/data_preprocess.py` (RGB resize to 480x640, JPEG output)
- [x] 1.2 Export GT 3D joints per sequence/sample in compact float16 format
- [x] 1.3 Support default output root `/opt/data/h36m_preprocessed` with configurable override

## 2. Loader/Docs

- [x] 2.1 Add or update H36M loader to read preprocessed SSD layout
- [x] 2.2 Document usage and output layout (include `uv run` command example)

## 3. Validation

- [x] 3.1 Run a small preprocessing dry-run on a subset to confirm output format
- [x] 3.2 Verify loader can read preprocessed samples

## 4. Training Config

- [x] 4.1 Add H36M FusionFormer config that uses preprocessed dataset root `/opt/data/h36m_preprocessed`
- [x] 4.2 Run a short `uv run` sanity check with the new config
