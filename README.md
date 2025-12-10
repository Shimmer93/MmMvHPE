# MMHPE
Multimodal Multiview Human Pose Estimation

## Installation

```bash
conda create -n mmhpe python=3.10 -y
conda activate mmhpe

conda install pytorch==2.2.0 torchvision==0.17.0 pytorch-cuda=11.8 lightning=2.1 -c pytorch -c conda-forge -c nvidia

pip install -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"

pip install git+https://github.com/state-spaces/mamba.git

pip install "chumpy==0.66"
mim install "mmpose==1.3.2"

cd models/pc_encoders/modules
python setup.py install // adjust cuda and gcc version if needed
cd ../../..

# pip install spconv-cu118
# mkdir third_party
# cd third_party
# git clone https://github.com/open-mmlab/OpenPCDet.git
# cd OpenPCDet
# pip install -r requirements.txt
# python setup.py develop
# cd ../..

pip install scikit-learn "numpy<2" "opencv-python<4.8" einops tensorboard tensorboardX cdflib wandb
```
