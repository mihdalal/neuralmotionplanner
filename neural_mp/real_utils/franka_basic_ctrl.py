"""
Franka control script that has integrated some basic control commands
"""

import numpy as np
from robofin.robots import FrankaRobot

from neural_mp.envs.franka_real_env import FrankaRealEnvManimo

env = FrankaRealEnvManimo(no_cam=True)

while True:
    input_str = input(
        "1. enter the name of the trajectory file to replay\n2. 'e' to exit\n3. 'reset' to set robot to home pos\n4. 'open' to open gripper\n5. 'close' to close gripper\n6. 'get_pose' to print end effector pose\n7. 'get_angles' to print joint angles\ninput: "
    )
    if input_str == "e":
        break
    elif input_str == "reset":
        env.reset()
    elif input_str == "open":
        env.step(gripper_action=1.0)
    elif input_str == "close":
        env.step(gripper_action=0.0)
    elif input_str == "get_pose":
        print("7D end effector pose. (xyz,xyzw): ", env.get_ee_pose())
    elif input_str == "get_angles":
        current_config = env.get_joint_angles()
        pose_mpinet = FrankaRobot.fk(current_config, eff_frame="right_gripper")
        print(f"current joint state: {current_config}")
        print(f"right gripper position: {pose_mpinet._xyz}")
        print(f"right gripper quaternion: {pose_mpinet._so3.wxyz}")
    else:
        file_path = "real_world_test_set/collected_trajs/" + input_str + ".npy"
        try:
            apply_traj = np.load(file_path)
        except:
            print("file not found")
            continue

        env.execute_plan(
            apply_traj,
            apply_traj[0],
            speed=0.2,
            set_intermediate_states=False,
            proprio_feedback=False,
            render=False,
        )
