# #! /bin/sh
eval "$(conda shell.bash hook)"
conda activate neural_mp
pip install -e pybullet-object-models/
pip install -e robomimic/
pip install -e pointnet2_ops/
pip install -e robofin/
pip install -e .
pip install pip==20.0.2
pip install setuptools==65.6.3
pip install wheel==0.38.4
pip install -r requirements.txt
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install "git+https://github.com/facebookresearch/pytorch3d.git"