import time
from collections import OrderedDict
from typing import Tuple

import meshcat
import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import torch
import torch._dynamo
import urchin
from robofin.pointcloud.torch import FrankaSampler
from robofin.robots import FrankaRobot

from neural_mp.envs.franka_real_env import FrankaRealEnv
from neural_mp.real_utils.homography_utils import save_pointcloud
from neural_mp.real_utils.model import NeuralMPModel


class NeuralMP:
    def __init__(
        self,
        env: FrankaRealEnv,
        model_url,
        train_mode,
        in_hand,
        in_hand_params=None,
        max_rollout_len=100,
        num_robot_points=2048,
        num_obstacle_points=4096,
        visualize=False,
    ):
        """
        Initialize the NeuralMP class.

        Args:
            env (FrankaRealEnv): robot control environment
            model_url (str): hugging face url to load the neural_mp model
            in_hand (bool, optional): Whether there is an object in hand
            in_hand_params (List[float], optional): The bounding box of the in hand object. 10 params in total [size(xyz), pos(xyz), ori(xyzw)] 3+3+4.
            max_rollout_len (int, optional): Maximum rollout length. Defaults to 100.
            num_robot_points (int, optional): Number of robot points in the input point cloud. Defaults to 2048.
            num_obstacle_points (int, optional): Number of obstacle points in the input point cloud. Defaults to 4096.
            visualize (bool, optional): Whether to visualize the evaluation with meshcat. Defaults to False.
        """
        # configuring PyTorch backend settings to optimize performance
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("medium")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch._dynamo.config.suppress_errors = True

        self.device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        self.policy = NeuralMPModel.from_pretrained(model_url)
        self.train_mode = train_mode
        self.env = env
        self.in_hand = in_hand
        self.in_hand_params = np.array(in_hand_params)
        self.env.collision_checker.set_cuboid_params(
            sizes=[[*self.in_hand_params[:3]]],
            centers=[[*self.in_hand_params[3:6]]],
            oris=[[*self.in_hand_params[6:]]],
        )
        self.max_rollout_len = max_rollout_len
        self.num_robot_points = num_robot_points
        self.num_obstacle_points = num_obstacle_points
        self.gpu_fk_sampler = FrankaSampler("cuda", use_cache=False)
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

    def get_scene_pcd(
        self,
        use_cache=False,
        cache_name=None,
        debug_raw_pcd=False,
        debug_combined_pcd=False,
        save_pcd=False,
        save_file_name="combined",
        filter=True,
        denoise=False,
    ):
        input("press Enter to collect pcd...")
        if use_cache:
            points = np.load("real_world_test_set/collected_pcds/" + cache_name + "_pcd.npy")
            colors = np.load("real_world_test_set/collected_pcds/" + cache_name + "_rgb.npy")
            if debug_combined_pcd:
                save_pointcloud(
                    "neural_mp/outputs/debug/combined_pcd.ply",
                    np.array(points),
                    np.array(colors)[:, [2, 1, 0]],
                )
                self.env.visualize_ply("neural_mp/outputs/debug/combined_pcd.ply")
        else:
            least_occlusion_config = np.array([0.0, -0.45, 0.0, -1.0, 0.0, 1.9, 0.7])
            self.env.move_robot_to_joint_state(joint_state=least_occlusion_config, time_to_go=4)
            points, colors = self.env.get_scene_pcd(
                debug_raw_pcd=debug_raw_pcd,
                debug_combined_pcd=debug_combined_pcd,
                save_pcd=save_pcd,
                save_file_name=save_file_name,
                filter=filter,
                denoise=denoise,
            )
            self.env.reset()
        return points, colors

    def prepare_point_cloud_for_inference(
        self,
        start_config: np.ndarray,
        goal_config: np.ndarray,
        obstacle_points: np.ndarray,
        obstacle_colors: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate the point cloud from the task specification.

        Args:
            start_config (np.ndarray): Joint angles of the robot at the start of the task.
            goal_config (np.ndarray): Joint angles of the robot at the goal of the task.
            obstacle_points (np.ndarray): xyz point coordinates of scene obstacles.
            obstacle_colors (np.ndarray): Colors corresponding to obstacle points.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: xyz and rgb information of the combined point cloud that will be passed into the network.
        """
        gripper_width = self.env.get_gripper_width() / 2
        start_tensor_config = torch.from_numpy(np.concatenate([start_config, [gripper_width]])).to(
            self.device
        )
        goal_tensor_config = torch.from_numpy(np.concatenate([goal_config, [gripper_width]])).to(
            self.device
        )
        robot_points = self.gpu_fk_sampler.sample(start_tensor_config, self.num_robot_points)
        target_points = self.gpu_fk_sampler.sample(goal_tensor_config, self.num_robot_points)

        if self.in_hand:
            robot_points = self.env.add_in_hand_pcd(
                robot_points, start_config[np.newaxis, :], self.in_hand_params
            )
            target_points = self.env.add_in_hand_pcd(
                target_points, goal_config[np.newaxis, :], self.in_hand_params
            )

        xyz = torch.cat(
            (
                torch.zeros(self.num_robot_points, 4, device=self.device),
                torch.ones(self.num_obstacle_points, 4, device=self.device),
                2 * torch.ones(self.num_robot_points, 4, device=self.device),
            ),
            dim=0,
        )
        xyz[: self.num_robot_points, :3] = robot_points.float()
        random_obstacle_indices = np.random.choice(
            len(obstacle_points), size=self.num_obstacle_points, replace=False
        )
        xyz[
            self.num_robot_points : self.num_robot_points + self.num_obstacle_points,
            :3,
        ] = torch.as_tensor(
            obstacle_points[random_obstacle_indices, :3], device=self.device
        ).float()
        xyz[
            self.num_robot_points + self.num_obstacle_points :,
            :3,
        ] = target_points.float()

        if self.visualize:
            len_obs = len(obstacle_colors)
            point_cloud_colors = np.zeros((3, len_obs + self.num_robot_points))
            point_cloud_colors[:, :len_obs] = obstacle_colors.T
            point_cloud_colors[0, len_obs:] = 1
            point_cloud_points = np.zeros((3, len_obs + self.num_robot_points))
            point_cloud_points[:, :len_obs] = obstacle_points.T
            point_cloud_points[:, len_obs:] = target_points[0].cpu().numpy().T
            self.viz["point_cloud"].set_object(
                # Don't visualize robot points
                meshcat.geometry.PointCloud(
                    position=point_cloud_points,
                    color=point_cloud_colors,
                    size=0.005,
                )
            )

            if self.in_hand:
                robot_pcd = robot_points.cpu().numpy()[0]
                robot_rgb = np.zeros((3, self.num_robot_points))
                robot_rgb[1, :] = 1
                self.viz["robot_point_cloud"].set_object(
                    # Don't visualize robot points
                    meshcat.geometry.PointCloud(
                        position=robot_pcd.T,
                        color=robot_rgb,
                        size=0.005,
                    )
                )
        obstacle_colors = obstacle_colors[random_obstacle_indices, :]
        return xyz, obstacle_colors

    @torch.no_grad()
    def motion_plan(self, start_config, goal_config, points, colors):
        """
        Motion plan by rolling out the policy.

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
        point_cloud = pset_points.unsqueeze(0)

        self.policy.start_episode()
        self.policy.policy.set_eval()

        q = torch.as_tensor(start_config, device=self.device).unsqueeze(0).float()
        goal_angles = torch.as_tensor(goal_config, device=self.device).unsqueeze(0).float()
        assert q.ndim == 2

        planning_success = False
        trajectory = [q]
        qt = q

        gripper_width = self.env.get_gripper_width() / 2

        def sampler(config, gripper_width=gripper_width):
            gripper_cfg = gripper_width * torch.ones((config.shape[0], 1), device=config.device)
            cfg = torch.cat((config, gripper_cfg), dim=1)
            if self.in_hand:
                sampled_pcd = self.gpu_fk_sampler.sample(cfg, self.num_robot_points)
                return self.env.add_in_hand_pcd(
                    sampled_pcd, config.cpu().numpy(), self.in_hand_params
                )
            else:
                return self.gpu_fk_sampler.sample(cfg, self.num_robot_points)

        obs = OrderedDict()
        obs["current_angles"] = q
        obs["goal_angles"] = goal_angles
        obs["compute_pcd_params"] = point_cloud

        ti0 = time.time()
        for i in range(self.max_rollout_len):
            qt = qt + self.policy.policy.get_action(obs_dict=obs)
            trajectory.append(qt)
            samples = sampler(qt).type_as(point_cloud)
            point_cloud[:, : samples.shape[1], :3] = samples
            obs["current_angles"] = qt
            obs["compute_pcd_params"] = point_cloud
        ti1 = time.time()
        ave_rollout_time = (ti1 - ti0) / self.max_rollout_len
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
                            robot_rgb = np.zeros((3, self.num_robot_points))
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

    @torch.no_grad()
    def motion_plan_with_tto(self, start_config, goal_config, points, colors, batch_size=100):
        """
        Motion plan by rolling out the policy with batched samples and perform test time optimization
        to select the safest path to execute on the robot.

        Args:
            points (np.ndarray): xyz information of the point cloud.
            colors (np.ndarray): rgb information of the point cloud for visualization.
            batch_size (int): size of the batch.

        Returns:
            Tuple[list, bool, float]: output trajectory, planning success flag, and average rollout time.
        """
        goal_pose = FrankaRobot.fk(goal_config, eff_frame="right_gripper")

        pset_points, obs_colors = self.prepare_point_cloud_for_inference(
            start_config, goal_config, points, colors
        )
        point_cloud = pset_points.unsqueeze(0).repeat(batch_size, 1, 1)

        self.policy.start_episode()
        if self.train_mode:
            self.policy.policy.set_train()
        else:
            self.policy.policy.set_eval()
        q = (
            torch.as_tensor(start_config, device=self.device)
            .unsqueeze(0)
            .float()
            .repeat(batch_size, 1)
        )
        goal_angles = (
            torch.as_tensor(goal_config, device=self.device)
            .unsqueeze(0)
            .float()
            .repeat(batch_size, 1)
        )
        assert q.ndim == 2

        planning_success = False
        trajectory = []
        qt = q

        gripper_width = self.env.get_gripper_width() / 2

        obs = OrderedDict()
        obs["current_angles"] = q
        obs["goal_angles"] = goal_angles
        obs["compute_pcd_params"] = point_cloud

        # limit max_rollout_len up to 100, so gpu memory does not explode
        max_rollout_len = min(self.max_rollout_len, 100)

        def sampler(config, gripper_width=gripper_width):
            gripper_cfg = gripper_width * torch.ones((config.shape[0], 1), device=config.device)
            cfg = torch.cat((config, gripper_cfg), dim=1)
            if self.in_hand:
                sampled_pcd = self.gpu_fk_sampler.sample(cfg, self.num_robot_points)
                return self.env.add_in_hand_pcd(
                    sampled_pcd, config.cpu().numpy(), self.in_hand_params
                )
            else:
                return self.gpu_fk_sampler.sample(cfg, self.num_robot_points)

        ti0 = time.time()
        for i in range(max_rollout_len):
            qt = qt + self.policy.policy.get_action(obs_dict=obs)
            trajectory.append(qt)
            samples = sampler(qt).type_as(point_cloud)
            point_cloud[:, : samples.shape[1], :3] = samples
            obs["current_angles"] = qt
            obs["compute_pcd_params"] = point_cloud

        ti1 = time.time()
        print(f"policy rollout time: {ti1 - ti0}")
        ti1 = time.time()

        output_traj = torch.stack(trajectory).permute(1, 0, 2)  # [batch_size, max_rollout_len, 7]
        goal_reaching = torch.norm(output_traj[:, -1] - goal_angles, dim=1) < 0.1
        output_traj = output_traj[goal_reaching]
        num_valid_traj = output_traj.shape[0]
        output_traj = output_traj.reshape(-1, 7)

        scene_pcd = point_cloud[
            :, self.num_robot_points : self.num_robot_points + self.num_obstacle_points, :3
        ][goal_reaching].repeat(max_rollout_len, 1, 1)
        waypoint_c_num = self.env.collision_checker.check_scene_collision_batch(
            output_traj, scene_pcd, thred=0.01, sphere_repr_only=(not self.in_hand)
        )
        traj_c_num = torch.sum(waypoint_c_num.reshape(num_valid_traj, max_rollout_len), dim=1)
        best_traj_idx = torch.argmin(traj_c_num)
        output_traj = (
            output_traj.reshape(num_valid_traj, max_rollout_len, -1)[best_traj_idx]
            .detach()
            .cpu()
            .numpy()
        )
        ti2 = time.time()
        ave_rollout_time = (ti2 - ti0) / max_rollout_len
        print(f"test time adaptation time: {ti2 - ti1}")
        print(
            f"The best trajectory out of {num_valid_traj}:\nnum_collisions: {traj_c_num[best_traj_idx]}\naverage num_collisions: {sum(traj_c_num)/num_valid_traj}"
        )

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
        output_traj = np.concatenate((start_config.reshape(1, 7), output_traj), axis=0)

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
                            robot_rgb = np.zeros((3, self.num_robot_points))
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

    def execute_motion_plan(self, plan, speed):
        """
        Execute a planned trajectory.

        Args:
            plan (list): List of joint angles along the path.
            speed (float): Speed for execution (rad/s).

        Returns:
            success (bool): check if the robot has reached the last state of the plan.
            joint_error (float): Error in joint angles between the last state and the target state.
        """
        success, joint_error, _ = self.env.execute_plan(
            plan=plan, init_joint_angles=plan[0], speed=speed
        )

        return success, joint_error
