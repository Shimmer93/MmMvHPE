# 1. Project Description
MMHPE (Multimodal Multiview Human Pose Estimation) is a Python codebase for training, evaluation, and inference of human pose models using multiple sensor modalities (for example RGB, depth, LiDAR, and mmWave).

- Main entrypoint: `main.py`
- Typical workflow: config-driven runs from `configs/` with outputs written to `logs/`

# 2. Python Environment
Use `uv` to manage dependencies and execution.

- Python: `>=3.12` (Linux environment)
- CUDA target in dependencies: `12.4`
- Install:
  - `uv sync`
- Run commands with `uv run`, for example:
  - `uv run python main.py ...`

Notes:
- This repository includes several GPU/CUDA-sensitive packages (`torch`, `spconv-cu124`, `flash-attn`, `mmcv`, etc.).
- If environment or CUDA assumptions change, update `pyproject.toml`. Do NOT touch `uv.lock`.

# 3. File Structure
Core directories and their roles:

- `main.py`: training/testing/prediction entrypoint
- `configs/`: YAML experiment configs
- `datasets/`: dataset classes and transforms
- `models/`: model architectures, heads, encoders, aggregators
- `losses/`, `metrics/`, `misc/`: training losses, evaluation metrics, utilities
- `tools/`: preprocessing, conversion, and helper scripts
- `third_party/`: vendored external projects/integrations
- `docs/`: project documentation
- `logs/`: outputs, checkpoints, and artifacts

# 4. Documentation
Within source codes, only put short and necessary comments. Instead, maintain a folder of documentation that:
- Describes the purpose of the codes, the high-level ideas of how they are implemented.
- Records the necessary details which can lead to confusions if missing.
- Gives a few useful examples. For scripts, these should be examples to call them in the command line. For code snippets, these should be examples to use and refer to them in other codes.
The structure of documentation are dynamic, one documentation file can focus on one source code file or multiple source code files, depending on the importance of the specific part. The golden standard of documentation is to let a new comer to quickly be ready for development.

# 5. Git
Only commit the changes when the user explicitly asks so. 

# 6. Current Pipeline (For RGB+PC-based HPE)
- Train the HPE model (Encoders + Aggregator + Keypoint Head + SMPL Head (Optional) )
- Train the Camera Head for PC
- Train the Camera Head for RGB
