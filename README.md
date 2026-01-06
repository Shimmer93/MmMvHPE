# MMHPE
Multimodal Multiview Human Pose Estimation

## Installation

1. Install `uv` if you have not done it:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    You may need to modify $PATH after installation.
2. You may want to install python using `uv`:
    ```
    uv python install 3.12
    ```
3. Run the following:
    ```bash
    uv sync
    uv pip install torch-scatter --force-reinstall
    ```

## Running

Append `uv run` to the front of your python commands.