# Neural MP: A Generalist Neural Motion Planner
This repository is the official implementation of Neural MP: A Generalist Neural Motion Planner

Neural MP is a machine learning-based motion planning system for robotic manipulation tasks. It combines neural networks trained on large-scale simulated data with lightweight optimization techniques to generate efficient, collision-free trajectories. Neural MP is designed to generalize across diverse environments and obstacle configurations, making it suitable for both simulated and real-world robotic applications. This repository contains the implementation, data generation tools, and evaluation scripts for Neural MP.


https://github.com/user-attachments/assets/17eeb664-ea4c-4904-b82e-cb5231fc84b9

**Authors**: [Murtaza Dalal*](https://mihdalal.github.io/), [Jiahui Yang*](https://jim-young6709.github.io/), [Russell Mendonca](https://russellmendonca.github.io/), [Youssef Khaky](https://www.linkedin.com/in/youssefkhaky/), [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/), [Deepak Pathak](https://www.cs.cmu.edu/~dpathak/)  
**Website**: [https://mihdalal.github.io/neuralmotionplanner](https://mihdalal.github.io/neuralmotionplanner)  
**Paper**: [https://mihdalal.github.io/neuralmotionplanner/resources/paper.pdf](https://mihdalal.github.io/neuralmotionplanner/resources/paper.pdf)  
**Models**: [https://huggingface.co/mihdalal/NeuralMP](https://huggingface.co/mihdalal/NeuralMP)  


If you find this codebase useful in your research, please cite:
```bibtex
@article{dalal2024neuralmp,
    title={Neural MP: A Generalist Neural Motion Planner},
    author={Murtaza Dalal and Jiahui Yang and Russell Mendonca and Youssef Khaky and Ruslan Salakhutdinov and Deepak Pathak},
    journal = {arXiv preprint arXiv:2409.05864},
    year={2024},
}
```

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation Instructions](#installation-instructions)
  - [1. Environment Setup](#1-environment-setup)
  - [2. System Dependencies](#2-system-dependencies)
  - [3. Clone Repository](#3-clone-repository)
  - [4. Python Dependencies](#4-python-dependencies)
  - [5. Real World Dependencies](#5-real-world-dependencies)
- [Real World Deployment](#real-world-deployment)
  - [Camera Calibration](#camera-calibration)
  - [Franka Basic Control Center](#franka-basic-control-center)
  - [Deploy Neural MP](#deploy-neural-mp)
  - [Real World Evaluations](#real-world-evaluations)
- [Coming Soon](#coming-soon)

## Prerequisites
- Conda
- NVIDIA GPU with appropriate drivers (for GPU support)

## Installation Instructions
### 1. Environment Setup
The system has been tested with: Python3.8, CUDA12.1, RTX3090 GPU with driver version 535

#### 1.1 Environment Variables
```bash
export PATH=/usr/local/cuda-12.1/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.1/
export WANDB_API_KEY=your_wandb_api_key_here
export MESA_GLSL_VERSION_OVERRIDE="330"
export MESA_GL_VERSION_OVERRIDE="3.3"
export OMP_NUM_THREADS=1  # Crucial for fast SubprocVecEnv performance!
```

#### 1.2 Create Conda Environment
```bash
conda create -n neural_mp python=3.8
conda activate neural_mp
```


### 2. System Dependencies
Install required system libraries:

```bash
sudo apt-get update && sudo apt-get install -y \
    swig cmake libgomp1 libjpeg8-dev zlib1g-dev libpython3.8 \
    libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev libegl1 \
    libglfw3-dev libglfw3 libgl1-mesa-glx libfdk-aac-dev libass-dev \
    libopus-dev libtheora-dev libvorbis-dev libvpx-dev libssl-dev \
    libboost-serialization-dev libboost-filesystem-dev libboost-system-dev \
    libboost-program-options-dev libboost-test-dev libeigen3-dev libode-dev \
    libyaml-cpp-dev libboost-python-dev libboost-numpy-dev libglfw3-dev \
    libgles2-mesa-dev patchelf libgl1-mesa-dev libgl1-mesa-glx libglew-dev \
    libosmesa6-dev
```

### 3. Clone Repository

To clone the repository with all its submodules, use the following command:

```bash
git clone --recurse-submodules https://github.com/mihdalal/neuralmotionplanner.git
cd neuralmotionplanner
```

If you've already cloned the repository without the `--recurse-submodules` flag, you can initialize and update the submodules like so:

```bash
git submodule update --init --recursive
```

Create directories to store real world data

```bash
mkdir real_world_test_set && cd real_world_test_set  && \
mkdir collected_configs collected_pcds collected_trajs evals && cd ..
```

### 4. Python Dependencies
#### 4.1 Set up OMPL
We provide two ways to install the [OMPL](https://ompl.kavrakilab.org/):
1. Build from source as appeared on the [official installation guide](https://ompl.kavrakilab.org/installation.html)
```bash
./install-ompl-ubuntu.sh --python
```
2. However, in case the compilation doesn't go through successfully, we provide a pre-compiled zip file as an alternate approach
```bash
unzip containers/ompl-1.5.2.zip
```

After installation, run
```
echo "<path to neuralmotionplanner folder>/neuralmotionplanner/ompl-1.5.2/py-bindings" >> ~/miniconda3/envs/neural_mp/lib/python3.8/site-packages/ompl.pth
```

#### 4.2 Install PyTorch and PyTorch3D
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

#### 4.3 Install Project Dependencies
```bash
pip install -e pybullet-object-models/
pip install -e robomimic/
pip install -e pointnet2_ops/
pip install -e robofin/
pip install -e ./
pip install -r requirements.txt
```

In practice we found that `pointnet2_ops` extension is not easy to build and often raise import errors. Hence, we offer our pre-compiled [`_ext.cpython-38-x86_64-linux-gnu.so`](https://github.com/mihdalal/pointnet2_ops/blob/v3.2.0/pointnet2_ops/_ext.cpython-38-x86_64-linux-gnu.so) file using python3.8, torch2.1.0 and CUDA12.1, so you can install the package without compiling new extensions. Please note, in order to use the pre-compiled extension, you need to comment out [this part of code](https://github.com/mihdalal/pointnet2_ops/blob/e9acd21c2da3bb803128ff1da1028bd2f377eb0e/setup.py#L23-L34) in setup.py, otherwise it will rebuild and overwrite the file.

#### 4.4 For contributers
```bash
pre-commit install
```

### 5. Real World Dependencies
If you are interested to run Neural MP in the real world, please install [ManiMo](https://github.com/mihdalal/manimo).
1. Set `MANIMO_PATH` as an environment variable in the `.bashrc` file:
   ```bash
   export MANIMO_PATH={FOLDER_PATH_TO_MANIMO}/manimo/manimo
   ```
2. Run the setup script on the client computer. Note that `mamba` setup does not work, always use `miniconda`:
   ```bash
   source setup_manimo_env_client.sh
   ```
3. Run the setup script on the server computer. Note that `mamba` setup does not work, always use `miniconda`:
   ```bash
   source setup_manimo_env_server.sh
   ```

To verify that the installation works, run the polymetis server on NUC by running the following script under the scripts folder:
```bash
python get_current_position.py
```


## Real World Deployment
Real world deployment commands on a setup of a single Franka robot with default panda gripper and multiple Intel Realsense Cameras

### Camera Calibration
#### Step 1: Setup camera id and intrinsics in ManiMo camera config at [multi_real_sense_neural_mp](https://github.com/mihdalal/manimo/blob/neural_mp/manimo/conf/camera/multi_real_sense_neural_mp.yaml)
1. Replace `device_id` with the actual camera id shown on its label
2. Execute script `get_camera_intrinsics.py`. Replace `intrinsics` with the terminal output.
```
python neural_mp/real_utils/get_camera_intrinsics.py -n <camera serial id>

e.g.
python neural_mp/real_utils/get_camera_intrinsics.py -n 102422076289
```
3. You may add/delete camera configs to accomendate the actual number of cameras you are using (in our setup we have 4 cameras). You can also adjust other camera parameters according to your need, but please follow the naming convention in the config file.

#### Step 2: Calibration with Apriltag
1. Print an Apriltag and attach it to the panda gripper, the larger the better, click here to visit the [April Tag generator](https://chaitanyantr.github.io/apriltag.html) (in our case we are using a `50mm` Apriltag from tag family `tagStandard52h13`, ID `17`)
2. Update the Apriltag specification into [calibration_apriltag.yaml](https://github.com/mihdalal/neural_mp/blob/master/neural_mp/configs/calibration_apriltag.yaml). You may set `display_images` to `True` to debug the images captured by the camera
3. Clear any potential obstacles in front of the robot.
4. Execute script `calibration_apriltag.py` for each camera, it will automatically calibrate camera extrinsics and save it as a `.pkl` file. Specify the index of the camera you want to calibrate and make sure the apriltag will be in view during calibration. You may activate the `--flip` flag to turn the end effector 180deg, so the apriltag could be captured by the cameras behind the robot.
```
# e.g. calibrate camera 1 which locates at the back of the robot 
python neural_mp/real_utils/calibration_apriltag.py -c 1 --flip
```
<div align="center">
  <img src="media/readme1.png" width="500" height="400" title="readme1">
</div>

#### Step 3: Manual Offset Tuning
In step 2, the calibration process will assume the center of the Apriltag locates exactly at the end effector position, which is normally not the case. There will exist a small xyz shift between the Apriltag and the actual end-effector position. To mitigate this error, we need to go through step 3 to update the `mv_shift` param in [multi_real_sense_neural_mp](https://github.com/mihdalal/manimo/blob/neural_mp/manimo/conf/camera/multi_real_sense_neural_mp.yaml).
1. Execute script `calibration_shift.py`
```
python neural_mp/real_utils/calibration_shift.py
```
2. Open the printed web link (should be something like `http://127.0.0.1:7000/static/`) for `meshcat` visualization
3. Now you should be able to see the point clouds captured by your cameras, as well as a yellow 'ground truth' robot in simulation. After the apriltag calibration, the captured point clouds should look almost correct, but with a small xyz shift. So now you should manually shift the point cloud so the robot points in the point cloud is overlapping with the 'ground truth' simulated robot. After you execute the script you will see the terminal guidance for this process. Once this is done, the script will print out the final xyz shift param, copy and paste them to replace `mv_shift` in [multi_real_sense_neural_mp](https://github.com/mihdalal/manimo/blob/neural_mp/manimo/conf/camera/multi_real_sense_neural_mp.yaml).

### Franka Basic Control Center
The script `franka_basic_ctrl.py` has integrated some basic control commands for the robot, such as `reset`, `open/close gripper`, `get current end effector pose / joint angles`, etc. After executing the script, you will see terminal guidance on how to run those commands.
```
python neural_mp/real_utils/franka_basic_ctrl.py
```

### Deploy Neural MP
We have uploaded our pre-trained model to hugging face, so you can easily load it by:
```
from neural_mp.real_utils.model import NeuralMPModel
neural_motion_planner = NeuralMPModel.from_pretrained("mihdalal/NeuralMP")
```
Note this only loads the base policy so test time optimization is not included.

To deploy Neural MP in the real world, please check the `NeuralMP` class in [neural_motion_planner.py](https://github.com/mihdalal/neural_mp/blob/master/neural_mp/real_utils/neural_motion_planner.py). We also provide a deployment example [deploy_neural_mp.py](https://github.com/mihdalal/neural_mp/blob/master/neural_mp/real_utils/deploy_neural_mp.py), which uses `NeuralMP` class with the Manimo control library. You may also use other Franka control libraries. Just create a wrapper class based on FrankaRealEnv, please follow the specified naming convensions.

Here's an example of running `deploy_neural_mp.py` with test time optimization and an in hand object. (size: 10cm x 10cm x 10cm ; relative pose to the end-effetor [xyz, xyzw] = [0, 0, 0.1, 0, 0, 0, 1])
```
python neural_mp/real_utils/deploy_neural_mp.py --tto --train-mode --in-hand --in-hand-params 0.1 0.1 0.1 0 0 0.1 0 0 0 1
```

Additionally you may use the `--mdl-url` argument to switch between different Neural_MP checkpoints. Now its default to `"mihdalal/NeuralMP"`.

### Real World Evaluations
For all the eval script we give detailed terminal guidance through out the whole process. We also have `meshcat` support to let you preview robot actions before real world execution. (again, you need to open the printed web link, should be something like `http://127.0.0.1:7000/static/`)
#### Collect Eval Configs
Manually collect target configurations for real world evaluation, the collected set will be saved in the output folder. Here is the detailed procedure:
1. Execute the script, and wait for it to initialize
```
python neural_mp/real_utils/collect_task_configs.py -n <config name>
```
2. Put franka in white mode, move robot to your desired configuration, then switch back to blue mode and reconnect the ManiMo controller
3. Enter 'y' in the terminal to save current config
4. Repeat step 2 and 3 until you collected all the config you need. Enter 'n' to exit

#### Neural MP Eval
There are two types of evaluations for Neural MP: motion planning with objects in hand and not in hand. Below shows the basic eval command
```
python neural_mp/real_evals/eval_neural_mp.py --mdl_url <hugging face url to the model repo> --cfg-set <name of the eval config> -l <name of the eval log file>
```

To execute a hand free eval with our provided checkpoint using test time optimization, you need to specify the `name of the eval config`, `name of the eval log file` and also turn on the relevant flags.
```
python neural_mp/real_evals/eval_neural_mp.py --cfg-set <name of the eval config> -l <name of the eval log file> --tto --train-mode
```

To execute an in-hand eval with our provided checkpoint using test time optimization, you need to turn on the `--in-hand` flag and also specify the bounding box of the in hand object. For `--in-hand-params`, you need to specify 10 params in total [size(xyz), pos(xyz), ori(xyzw)] 3+3+4.
```
python neural_mp/real_evals/eval_neural_mp.py --cfg-set <name of the eval config> -l <name of the eval log file> --tto --train-mode --in-hand --in-hand-param <geometric info>
```

We also have some useful debugging flags for the eval, such as `--debug-combined-pcd`. Once turned on, it will show you the visualization of the combined point cloud captured by the cameras.

To see all of the options available, run
```
python neural_mp/real_evals/eval_neural_mp.py --help
```

#### Baseline Evals
MPiNets (Motion Policy Networks)
```
python neural_mp/real_evals/eval_mpinet.py --mdl-path <MPiNets ckpt path> --cfg-set <name of the eval config> -l <name of the eval log file>
```

AITStar
```
python neural_mp/real_evals/eval_aitstar.py <planning time> <number of waypoints in the trajectory> --cfg-set <name of the eval config> -l <name of the eval log file>

# in our case, we were using 10s and 80s as the planning time and 50 as the number of trajectory waypoints
```

# Coming Soon
Please stay tuned, we will soon be releasing the full dataset used to train our model (1M simulated trajectories), the data-generation code, the training code and dockerfiles to make the entire running process much smoother.

## Citation

Please cite [the Neural MP paper](https://mihdalal.github.io/neuralmotionplanner/resources/paper.pdf) if you use this code in your work:

```bibtex
@article{dalal2024neuralmp,
    title={Neural MP: A Generalist Neural Motion Planner},
    author={Murtaza Dalal and Jiahui Yang and Russell Mendonca and Youssef Khaky and Ruslan Salakhutdinov and Deepak Pathak},
    journal = {arXiv preprint arXiv:2409.05864},
    year={2024},
}
```
