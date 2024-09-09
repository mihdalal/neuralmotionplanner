"""
manually collect goal configurations for real world evaluation
"""

import argparse

import numpy as np
from robofin.robots import FrankaRobot

from neural_mp.envs.franka_real_env import FrankaRealEnvManimo

parser = argparse.ArgumentParser()
parser.add_argument(
    "-n",
    "--config-name",
    required=True,
    help="name of the config file to save",
)
args = parser.parse_args()

env = FrankaRealEnvManimo()

configs = []

while True:
    record = input("record position? (y/n): ")

    if record == "y":
        current_config = env.get_joint_angles()
        pose_mpinet = FrankaRobot.fk(current_config, eff_frame="right_gripper")
        print(f"recorded joint state: {current_config}")
        print(f"right gripper position: {pose_mpinet._xyz}")
        print(f"right gripper quaternion: {pose_mpinet._so3.wxyz}")

        configs.append(current_config)
    elif record == "n":
        break

configs = np.array(configs)
np.save("real_world_test_set/collected_configs/" + args.config_name + ".npy", configs)
