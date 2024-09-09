"""
AITstar evaluation
"""


import argparse
import time
from typing import Tuple

import meshcat
import numpy as np
import torch
from robofin.robots import FrankaRobot

from neural_mp.envs.franka_real_env import FrankaRealEnvManimo
from neural_mp.real_evals.eval_base import EvalBase, rw_eval
from neural_mp.real_utils.homography_utils import save_pointcloud

NUM_ROBOT_POINTS = 2048
NUM_OBSTACLE_POINTS = 4096
NUM_TARGET_POINTS = 128
MAX_ROLLOUT_LENGTH = 150


class EvalAITstar(EvalBase):
    def __init__(
        self, env: FrankaRealEnvManimo, planning_time=1, num_waypoints=50, visualize=False
    ):
        """
        Initialize the EvalAITstar class.

        Args:
            env (FrankaRealEnvManimo): Real world deployment environment to evaluate the agent.
            planning_time (float): Time allocated for planning.
            num_waypoints (int): Number of waypoints in the output trajectory.
            visualize (bool): Whether to visualize the evaluation with meshcat.
        """
        super().__init__(env, visualize)
        self.planning_time = planning_time
        self.num_waypoints = num_waypoints

    def prepare_point_cloud_for_inference(
        self,
        start_config: np.ndarray,
        goal_config: np.ndarray,
        obstacle_points: np.ndarray,
        obstacle_colors: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate the point cloud from the eval task specification.

        Args:
            start_config (np.ndarray): Joint angles of the robot at the start of the task.
            goal_config (np.ndarray): Joint angles of the robot at the goal of the task.
            obstacle_points (np.ndarray): xyz point coordinates of scene obstacles.
            obstacle_colors (np.ndarray): Colors corresponding to obstacle points.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: xyz and rgb information of the combined point cloud that will be passed into the network.
        """
        gripper_width = self.env.get_gripper_width() / 2
        robot_points = self.cpu_fk_sampler.sample(
            torch.Tensor([*start_config, gripper_width]), NUM_ROBOT_POINTS
        )
        target_points = self.cpu_fk_sampler.sample(
            torch.Tensor([*goal_config, gripper_width]), NUM_ROBOT_POINTS
        )

        xyz = torch.cat(
            (
                torch.zeros(NUM_ROBOT_POINTS, 4),
                torch.ones(NUM_OBSTACLE_POINTS, 4),
                2 * torch.ones(NUM_ROBOT_POINTS, 4),
            ),
            dim=0,
        )
        xyz[:NUM_ROBOT_POINTS, :3] = robot_points.float()
        random_obstacle_indices = np.random.choice(
            len(obstacle_points), size=NUM_OBSTACLE_POINTS, replace=False
        )
        xyz[
            NUM_ROBOT_POINTS : NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS,
            :3,
        ] = torch.as_tensor(obstacle_points[random_obstacle_indices, :3]).float()
        xyz[
            NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS :,
            :3,
        ] = target_points.float()
        obstacle_colors = obstacle_colors[random_obstacle_indices, :]

        if self.visualize:
            point_cloud_colors = np.zeros((3, NUM_OBSTACLE_POINTS + NUM_ROBOT_POINTS))
            point_cloud_colors[:, :NUM_OBSTACLE_POINTS] = obstacle_colors.T
            point_cloud_colors[0, NUM_OBSTACLE_POINTS:] = 1
            self.viz["point_cloud"].set_object(
                # Don't visualize robot points
                meshcat.geometry.PointCloud(
                    position=xyz[NUM_ROBOT_POINTS:, :3].numpy().T,
                    color=point_cloud_colors,
                    size=0.005,
                )
            )
        return xyz, obstacle_colors

    def motion_plan(self, start_config, goal_config, points, colors):
        """
        motion planning using AITstar.

        Args:
            start_config (np.ndarray): Joint angles of the robot at the start of the task.
            goal_config (np.ndarray): Joint angles of the robot at the goal of the task.
            points (np.ndarray): xyz information of the point cloud.
            colors (np.ndarray): rgb information of the point cloud for visualization.

        Returns:
            Tuple[list, bool, float]: output trajectory, planning success flag, and average rollout time.
        """
        goal_pose = FrankaRobot.fk(goal_config, eff_frame="right_gripper")

        obs_points, obs_colors = self.prepare_point_cloud_for_inference(
            start_config, goal_config, points, colors
        )
        obs_points = obs_points[NUM_ROBOT_POINTS : NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS, :3]

        planning_success = False
        gripper_width = self.env.get_gripper_width() / 2

        ti0 = time.time()
        output_traj = self.env.mp_to_joint_target(
            start_angles=start_config,
            target_angles=goal_config,
            obstacle_points=obs_points,
            planning_time=self.planning_time,
            num_waypoints=self.num_waypoints,
        )
        ti1 = time.time()
        planning_time = ti1 - ti0
        print(f"planning time: {ti1 - ti0}")

        # check whether goal is reached
        for i in range(len(output_traj)):
            eff_pose = FrankaRobot.fk(output_traj[i], eff_frame="right_gripper")
            pos_err = np.linalg.norm(eff_pose._xyz - goal_pose._xyz)
            ori_err = np.abs(
                np.degrees((eff_pose.so3._quat * goal_pose.so3._quat.conjugate).radians)
            )

            if (
                np.linalg.norm(eff_pose._xyz - goal_pose._xyz) < 0.01
                and np.abs(np.degrees((eff_pose.so3._quat * goal_pose.so3._quat.conjugate).radians))
                < 15
            ):
                planning_success = True

        print(f"sim results:\nstep: {i+1}\npos_err: {pos_err*100} cm\nori_err: {ori_err} deg")

        if self.visualize:
            trajc = output_traj.copy()
            while True:
                visual = input("simulate trajectory? (y/n): ")
                if visual == "n":
                    break
                elif visual == "y":
                    print("simlating")
                    for idx in range(len(trajc)):
                        sim_config = np.append(trajc[idx], gripper_width)
                        for idx, (k, v) in enumerate(
                            self.urdf.visual_trimesh_fk(sim_config[:8]).items()
                        ):
                            self.viz[f"robot/{idx}"].set_transform(v)
                        time.sleep(0.08)

        return output_traj, planning_success, planning_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("planning_time", type=int, help="Time for running AITstar")
    parser.add_argument(
        "num_waypoints", type=int, help="Number of waypoints for the output trajectory"
    )
    parser.add_argument(
        "--cfg-set",
        type=str,
        default="scene1_l1",
        help="Specify the config set for testing",
    )
    parser.add_argument(
        "--cache-name",
        type=str,
        default="scene1_single_block",
        help="Specify the scene cache file with pcd and rgb data",
    )
    parser.add_argument(
        "--cam-only",
        action="store_true",
        help=(
            "If set, franka_env will launch in cam_only mode, where the robot will not be connected"
        ),
    )
    parser.add_argument(
        "--arm-only",
        action="store_true",
        help=(
            "If set, robot will launch in arm_only mode, where the gripper will not be connected"
        ),
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help=("If set, will use pre-stored point clouds"),
    )
    parser.add_argument(
        "--debug-raw-pcd",
        action="store_true",
        help=("If set, will show visualization of raw pcd from each camera"),
    )
    parser.add_argument(
        "--debug-combined-pcd",
        action="store_true",
        help=("If set, will show visualization of the combined pcd"),
    )
    parser.add_argument(
        "--denoise-pcd",
        action="store_true",
        help=("If set, will apply denoising to the pcds"),
    )
    parser.add_argument(
        "--save-pcd",
        action="store_true",
        help=("If set, will save the combined pcd"),
    )
    parser.add_argument(
        "-l",
        "--log-name",
        type=str,
        default="rw_eval_aitstar",
        help="Specify the file name for logging eval info",
    )
    parser.add_argument(
        "--in-hand", action="store_true", help=("If set, will apply inference time optimization")
    )
    parser.add_argument(
        "--tto", action="store_true", help=("If set, will apply test time optimization")
    )
    args = parser.parse_args()

    env = FrankaRealEnvManimo(cam_only=args.cam_only, arm_only=args.arm_only)

    # get pcd
    input("press Enter to collect pcd...")
    if args.use_cache:
        points = np.load("real_world_test_set/collected_pcds/" + args.cache_name + "_pcd.npy")
        colors = np.load("real_world_test_set/collected_pcds/" + args.cache_name + "_rgb.npy")
        if args.debug_combined_pcd:
            save_pointcloud(
                "neural_mp/outputs/debug/combined_pcd.ply",
                np.array(points),
                np.array(colors)[:, [2, 1, 0]],
            )
            env.visualize_ply("neural_mp/outputs/debug/combined_pcd.ply")
    else:
        least_occlusion_config = np.array([0.0, -0.45, 0.0, -1.0, 0.0, 1.9, 0.7])
        env.move_robot_to_joint_state(joint_state=least_occlusion_config, time_to_go=4)
        points, colors = env.get_scene_pcd(
            debug_raw_pcd=args.debug_raw_pcd,
            debug_combined_pcd=args.debug_combined_pcd,
            save_pcd=args.save_pcd,
            denoise=args.denoise_pcd,
        )
        env.reset()

    # init test
    eval_agent = EvalAITstar(
        env=env, planning_time=args.planning_time, num_waypoints=args.num_waypoints, visualize=True
    )
    config_set = np.load("real_world_test_set/collected_configs/" + args.cfg_set + ".npy")

    rw_eval(
        eval_agent=eval_agent,
        env=env,
        points=points,
        colors=colors[:, [2, 1, 0]],
        config_set=config_set,
        arm_only=args.arm_only,
        args=args,
        file_path="./real_world_test_set/evals/" + args.log_name + ".csv",
    )
