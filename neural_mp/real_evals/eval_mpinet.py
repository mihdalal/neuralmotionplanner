"""
MPiNets evaluation
"""

import argparse
import time
from typing import Tuple

import meshcat
import numpy as np
import torch
from mpinets.model import MotionPolicyNetwork
from mpinets.utils import normalize_franka_joints, unnormalize_franka_joints
from robofin.robots import FrankaRobot

from neural_mp.envs.franka_real_env import FrankaRealEnvManimo
from neural_mp.real_evals.eval_base import EvalBase, rw_eval
from neural_mp.real_utils.homography_utils import save_pointcloud

NUM_ROBOT_POINTS = 2048
NUM_OBSTACLE_POINTS = 4096
NUM_TARGET_POINTS = 128
MAX_ROLLOUT_LENGTH = 100


class EvalMPiNets(EvalBase):
    def __init__(self, args, env: FrankaRealEnvManimo, visualize=False):
        """
        Initialize the EvalMPiNets class.

        Args:
            env (FrankaRealEnvManimo): Real world deployment environment to evaluate the agent.
            visualize (bool): Whether to visualize the evaluation with meshcat.
        """
        super().__init__(env, visualize)
        self.mdl = MotionPolicyNetwork.load_from_checkpoint(args.mdl_path).cuda()
        self.mdl.eval()
        self.in_hand_params = np.array(args.in_hand_params)
        self.in_hand = args.in_hand

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
        goal_pose = FrankaRobot.fk(goal_config, eff_frame="right_gripper")

        gripper_width = self.env.get_gripper_width() / 2
        robot_points = self.cpu_fk_sampler.sample(
            torch.Tensor([*start_config, gripper_width]), NUM_ROBOT_POINTS
        )
        target_points = self.cpu_fk_sampler.sample_end_effector(
            torch.as_tensor(goal_pose.matrix).type_as(robot_points).unsqueeze(0),
            num_points=NUM_TARGET_POINTS,
            gripper_width=gripper_width,
        )

        robot_points = robot_points.cuda()
        target_points = target_points.cuda()

        if self.in_hand:
            robot_points = self.env.add_in_hand_pcd(
                robot_points, start_config[np.newaxis, :], self.in_hand_params
            ).cpu()
            target_points = self.env.add_in_hand_pcd(
                target_points,
                goal_config[np.newaxis, :],
                self.in_hand_params,
                num_in_hand_points=100,
            ).cpu()

        xyz = torch.cat(
            (
                torch.zeros(NUM_ROBOT_POINTS, 4),
                torch.ones(NUM_OBSTACLE_POINTS, 4),
                2 * torch.ones(NUM_TARGET_POINTS, 4),
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
            point_cloud_colors = np.zeros((3, NUM_OBSTACLE_POINTS + NUM_TARGET_POINTS))
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

    @torch.no_grad()
    def motion_plan(self, start_config, goal_config, points, colors):
        """
        motion planning with MPiNets.

        Args:
            start_config (np.ndarray): Joint angles of the robot at the start of the task.
            goal_config (np.ndarray): Joint angles of the robot at the goal of the task.
            points (np.ndarray): xyz information of the point cloud.
            colors (np.ndarray): rgb information of the point cloud for visualization.

        Returns:
            Tuple[list, bool, float]: output trajectory, planning success flag, and average rollout time.
        """
        goal_pose = FrankaRobot.fk(goal_config, eff_frame="right_gripper")

        pset_points, obs_colors = self.prepare_point_cloud_for_inference(
            start_config, goal_config, points, colors
        )
        point_cloud = pset_points.unsqueeze(0).cuda()

        q = torch.as_tensor(start_config).unsqueeze(0).float().cuda()
        assert q.ndim == 2

        planning_success = False
        trajectory = [q]
        q_norm = normalize_franka_joints(q)
        assert isinstance(q_norm, torch.Tensor)

        gripper_width = self.env.get_gripper_width() / 2

        def sampler(config, gripper_width=gripper_width):
            """
            Sample the point cloud based on eval configuration.

            Args:
                config (torch.Tensor): Configuration to sample.
                gripper_width (float): Width of the gripper.

            Returns:
                torch.Tensor: Sampled point cloud.
            """
            gripper_cfg = gripper_width * torch.ones((config.shape[0], 1), device=config.device)
            cfg = torch.cat((config, gripper_cfg), dim=1)
            if self.in_hand:
                sampled_pcd = self.gpu_fk_sampler.sample(cfg, NUM_ROBOT_POINTS)
                return self.env.add_in_hand_pcd(
                    sampled_pcd, config.cpu().numpy(), self.in_hand_params
                )
            else:
                return self.gpu_fk_sampler.sample(cfg, NUM_ROBOT_POINTS)

        ti0 = time.time()
        for i in range(MAX_ROLLOUT_LENGTH):
            q_norm = torch.clamp(q_norm + self.mdl(point_cloud, q_norm), min=-1, max=1)
            qt = unnormalize_franka_joints(q_norm)
            assert isinstance(qt, torch.Tensor)
            trajectory.append(qt)

            samples = sampler(qt).type_as(point_cloud)
            point_cloud[:, : samples.shape[1], :3] = samples
        ti1 = time.time()
        ave_rollout_time = (ti1 - ti0) / MAX_ROLLOUT_LENGTH
        print(f"policy rollout time: {ti1 - ti0}")

        output_traj = [t.squeeze().detach().cpu().numpy() for t in trajectory]
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
                output_traj = output_traj[: (i + 1)]
                break

        print(f"sim results:\nstep: {i+1}\npos_err: {pos_err*100} cm\nori_err: {ori_err} deg")

        if self.visualize:
            trajc = output_traj.copy()
            while True:
                visual = input("simulate trajectory? (y/n): ")
                if visual == "n":
                    break
                elif visual == "y":
                    print("simlating")
                    for idx_traj in range(len(trajc)):
                        sim_config = np.append(trajc[idx_traj], gripper_width)
                        for idx, (k, v) in enumerate(
                            self.urdf.visual_trimesh_fk(sim_config[:8]).items()
                        ):
                            self.viz[f"robot/{idx}"].set_transform(v)

                        if self.in_hand:
                            # visualize robot pcd as well
                            robot_pcd = (
                                sampler(torch.Tensor(trajc[idx_traj]).cuda().unsqueeze(0))
                                .cpu()
                                .numpy()[0]
                            )
                            robot_rgb = np.zeros((3, NUM_ROBOT_POINTS))
                            robot_rgb[1, :] = 1
                            self.viz["robot_point_cloud"].set_object(
                                # Don't visualize robot points
                                meshcat.geometry.PointCloud(
                                    position=robot_pcd.T,
                                    color=robot_rgb,
                                    size=0.005,
                                )
                            )
                        time.sleep(0.05)

        return output_traj, planning_success, ave_rollout_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mdl-path",
        type=str,
        default="../motion-policy-networks/mpinets_hybrid_expert.ckpt",
        help="A checkpoint file from MPiNets",
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
        default="rw_eval_mpinet",
        help="Specify the file name for logging eval info",
    )
    parser.add_argument(
        "--in-hand", action="store_true", help=("If set, will enable in hand mode for eval")
    )
    parser.add_argument(
        "--in-hand-params",
        nargs="+",
        type=float,
        default=[0.26, 0.08, 0.2, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 1.0],
        help="Specify the bounding box of the in hand object. 10 params in total [size(xyz), pos(xyz), ori(xyzw)] 3+3+4.",
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

    # init eval
    eval_agent = EvalMPiNets(args=args, env=env, visualize=True)
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
