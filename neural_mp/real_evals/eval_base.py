"""
Base environment for real world evaluation
"""

import csv
import datetime
from abc import abstractmethod

import grpc
import meshcat
import numpy as np
import urchin
from robofin.pointcloud.torch import FrankaSampler
from robofin.robots import FrankaRobot
from tqdm import tqdm


def rw_eval(
    eval_agent,
    env,
    points,
    colors,
    config_set,
    arm_only,
    args,
    file_path="./real_world_test_set/evals/rw_eval.csv",
):
    """
    Evaluate the agent in the real world environment.

    Args:
        eval_agent: Agent to evaluate.
        env: Real world deployment environment to evaluate the agent.
        points (np.ndarray): Point cloud points.
        colors (np.ndarray): Point cloud colors.
        config_set (list): Set of configurations to evaluate.
        arm_only (bool): Whether to use only the arm. (launch without gripper)
        args: Additional arguments for the evaluation.
        file_path (str): Path to the CSV file for logging the evaluation.
    """
    file = open(file_path, mode="w", newline="")
    writer = csv.writer(file)
    writer.writerow(
        [
            "index",
            "start_time",
            "end_time",
            "planning_success",
            "success",
            "pos_err (cm)",
            "ori_err (deg)",
            "has_physical_vio (collision -> 1)",
            "rollout_time_per_step (s)",
        ]
    )
    planning_success_l = []
    success_l = []
    pos_err_l = []
    ori_err_l = []
    vios_l = []
    rollout_step_time_l = []

    for i in range(len(config_set) + 1):
        print(f"\ntesting #{i+1}:")
        input("press Enter to start planning...")
        if i == 0:
            while True:
                plan_start = input("at start state, type y to plan and n to skip (y/n): ")
                if plan_start in ["y", "n"]:
                    break
            if plan_start == "y":
                start_config = env.get_joint_angles()
                goal_config = config_set[i]
            elif plan_start == "n":
                continue
        elif i == len(config_set):
            while True:
                plan_start = input("at start state, type y to plan and n to skip (y/n): ")
                if plan_start in ["y", "n"]:
                    break
            if plan_start == "y":
                start_config = config_set[i - 1]
                goal_config = env.canonical_joint_pose
            elif plan_start == "n":
                continue
        else:
            start_config = config_set[i - 1]
            goal_config = config_set[i]

        # rollout
        exe_success = False
        while not exe_success:
            if args.tto:
                batch_size = 100
                (
                    apply_traj,
                    planning_success,
                    rollout_step_time,
                ) = eval_agent.motion_plan_with_tto(
                    start_config, goal_config, points, colors, batch_size
                )
            else:
                apply_traj, planning_success, rollout_step_time = eval_agent.motion_plan(
                    start_config, goal_config, points, colors
                )

            while True:
                isreplan = input("y to execute and n to replan: ")
                if isreplan in ["y", "n"]:
                    break
            if isreplan == "n":
                continue

            # execute
            try:
                start_time = datetime.datetime.now()
                exe_success, exe_joint_error, frames = env.execute_plan(
                    apply_traj,
                    apply_traj[0],
                    speed=0.2,
                    proprio_feedback=False,
                    render=False,
                )
                success, pos_err, ori_err = env.get_success(goal_config)

            except grpc._channel._InactiveRpcError:
                input("press Enter to reconnect robot, make sure server is launched...")
                import ipdb

                ipdb.set_trace()
                env.robot_reconnect(arm_only=arm_only)
                exe_success = False
                while True:
                    isreplan = input(
                        "execution stopped by user, type redo to re-execute the plan, type skip to go to the next task: "
                    )
                    if isreplan in ["redo", "skip"]:
                        break
                if isreplan == "redo":
                    idx = i - 1
                else:
                    idx = i
                    exe_success = True
                    success, pos_err, ori_err = env.get_success(eval_agent.goal_config)

                input(
                    "press Enter to set robot to the starting config, note this process doesn't guarantee to be collision free..."
                )
                if idx == len(config_set):
                    if args.in_hand:
                        input(
                            "will reset robot, should first release in hand obj, press Enter to open the gripper..."
                        )
                        env.set_gripper_state(gripper_width=1.0)
                    input("press Enter to reset...")
                    env.reset()
                else:
                    env.move_robot_to_joint_state(joint_state=config_set[idx], time_to_go=4)

            print(f"eval results:\npos_err: {pos_err} cm\nori_err: {ori_err} deg")

        while True:
            if i == 0:
                print(
                    "Note: the first start_config is the start pos of the robot, you might want to skip this eval"
                )
            elif i == len(config_set):
                print(
                    "Note: the last goal_config is the canonical joint pose of the robot, you might want to skip this eval"
                )
            vio_info = input(
                "\nrecord physical vio info, type y for violations and s for skips (y/n/s)"
            )
            if vio_info in ["y", "n", "s"]:
                break

        end_time = datetime.datetime.now()
        np.save(
            "real_world_test_set/collected_trajs/" + f"{args.log_name}{i}_{end_time}" + ".npy",
            apply_traj,
        )

        writer.writerow(
            [
                i + 1,
                f"{start_time}",
                f"{end_time}",
                planning_success,
                success,
                pos_err,
                ori_err,
                vio_info,
                rollout_step_time,
            ]
        )
        if vio_info == "s":
            continue
        planning_success_l.append(int(planning_success))
        success_l.append(int(success))
        pos_err_l.append(pos_err)
        ori_err_l.append(ori_err)
        if vio_info == "y":
            vios_l.append(1)
        elif vio_info == "n":
            vios_l.append(0)
        rollout_step_time_l.append(rollout_step_time)

    writer.writerow(
        [
            "average",
            "N/A",
            "N/A",
            np.mean(planning_success_l),
            np.mean(success_l),
            np.mean(pos_err_l),
            np.mean(ori_err_l),
            np.mean(vios_l),
            np.mean(rollout_step_time_l),
        ]
    )
    file.close()

    print("testing finished")
    input("press Enter to reset...")
    env.reset()


class EvalBase:
    def __init__(self, env, visualize=False):
        """
        Initialize the evaluation base class.

        Args:
            env: Real world deployment environment to evaluate the agent.
            visualize (bool): Whether to visualize the evaluation with meshcat.
        """
        self.env = env
        self.visualize = visualize
        if self.visualize:
            self.viz = meshcat.Visualizer()
            # Load the FK module
            self.urdf = urchin.URDF.load(FrankaRobot.urdf)
            # Preload the robot meshes in meshcat at a neutral position
            for idx, (k, v) in enumerate(self.urdf.visual_trimesh_fk(np.zeros(8)).items()):
                self.viz[f"robot/{idx}"].set_object(
                    meshcat.geometry.TriangularMeshGeometry(k.vertices, k.faces),
                    meshcat.geometry.MeshLambertMaterial(wireframe=False),
                )
                self.viz[f"robot/{idx}"].set_transform(v)

        self.cpu_fk_sampler = FrankaSampler("cpu", use_cache=True)
        self.gpu_fk_sampler = FrankaSampler("cuda:0", use_cache=True)
        self.setup_configs()

    def setup_configs(self, start_config=None, goal_config=None):
        """
        Setup start and goal configurations.

        Args:
            start_config (np.ndarray, optional): Start joint configuration.
            goal_config (np.ndarray, optional): Goal joint configuration.
        """
        self.start_config = (
            np.array(
                [0.5069095, 0.48568127, -0.10151276, -1.9154993, 0.00240945, 2.3056712, 1.0592909]
            )
            if start_config is None
            else start_config
        )

        self.goal_config = (
            np.array(
                [-0.6083077, 0.34265295, 0.10220415, -1.9886992, -0.14613454, 2.422832, -2.568265]
            )
            if goal_config is None
            else goal_config
        )

        self.goal_pose = FrankaRobot.fk(self.goal_config, eff_frame="right_gripper")

    @abstractmethod
    def make_point_cloud_from_problem(self):
        """
        Generate corresponding point cloud for the tasks and visualization
        """
        pass

    @abstractmethod
    def rollout_until_success(self):
        """
        rollout the policy until success or MAX_ROLLOUT_LENGTH
        """
        pass
