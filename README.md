# MMHPE
Multimodal Multiview Human Pose Estimation

## Installation

```bash
mamba create -n mmhpe python=3.12 -y
mamba activate mmhpe

mamba install pytorch=2.5.1 torchvision=0.20.1 lightning uv -c conda-forge

uv pip install scikit-learn einops tensorboard tensorboardX cdflib wandb rerun-sdk smplpytorch timm ultralytics rich opencv-python
uv pip install --no-build-isolation git+https://github.com/mattloper/chumpy
uv pip install git+https://github.com/facebookresearch/segment-anything.git
uv pip install git+https://github.com/state-spaces/mamba.git # not needed for now, takes quite a long time to install

cd models/pc_encoders/modules
python setup.py install # adjust cuda and gcc version if needed
cd ../../..
```
