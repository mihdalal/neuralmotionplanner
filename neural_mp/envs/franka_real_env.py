"""
This file contains the mp_env wrapper that is used to provide
a standardized environment API for real world evaluation and 
testing purposes. In order to utilize the wrapper env, pre-installation
of manimo (https://github.com/AGI-Labs/manimo.git) is required
"""

import datetime
import faulthandler
import time
from abc import abstractmethod
from collections import OrderedDict
from typing import Tuple

import cv2
import hydra
import numpy as np
import open3d as o3d
import pyquaternion
import torch
import urchin
from atob.caelan_smoothing import smooth_cubic
from manimo.actuators.arms.franka_arm import FrankaArm
from manimo.actuators.grippers.polymetis_gripper import PolymetisGripper
from manimo.environments.single_arm_env import SingleArmEnv
from manimo.utils.helpers import Rate
from ompl import base as ob
from ompl import geometric as og
from pyquaternion import Quaternion
from robofin.pointcloud.torch import FrankaSampler
from robofin.robots import FrankaRobot
from scipy.spatial.transform import Rotation as R_scipy
from torchcontrol.transform import Rotation as R
from torchcontrol.transform import Transformation as T

from neural_mp.real_utils.homography_utils import HomographyTransform, save_pointcloud
from neural_mp.real_utils.real_world_collision_checker import FrankaCollisionChecker
from neural_mp.utils.constants import (
    FRANKA_ACCELERATION_LIMIT,
    FRANKA_LOWER_LIMITS,
    FRANKA_UPPER_LIMITS,
    FRANKA_VELOCITY_LIMIT,
)
from neural_mp.utils.geometry import (
    TorchCuboids,
    TorchCylinders,
    TorchSpheres,
    vectorized_subsample,
)


def crop_and_resize_back_view(img, output_size=(84, 84)):
    """
    Crop and resize the view of an image.

    Args:
        img (np.ndarray): The input image.
        output_size (tuple): The desired output size (width, height).

    Returns:
        np.ndarray: The cropped and resized image.
    """
    out = cv2.resize(img[100:300, 150:-100], output_size)
    return out


class FrankaRealEnv:
    """
    Base class for controlling the Franka robot in the real world. Implement the abstract functions with your robot control library.
    """

    def __init__(self):
        self.ctrl_hz = ...
        self.collision_checker = FrankaCollisionChecker()

    @abstractmethod
    def get_joint_angles(self):
        """
        Get the joint angles of the robot.
        Returns:
            joint_angles (np.ndarray): 7-dof joint angles.
        """

    @abstractmethod
    def get_gripper_width(self):
        """
        Get the gripper width.

        Returns:
            float: Gripper width.
        """

    @abstractmethod
    def reset(self):
        """
        Reset joint angles to canonical pose.
        """

    @abstractmethod
    def step(self, joint_action: np.ndarray = None, gripper_action: float = None):
        """
        Step the robot to the target joint state without any interpolation.
        Be careful about your step size, large joint displacement might hurt the robot motor.

        Args:
            joint_action (np.ndarray): Action to take for the arm. (absolute joint angles)
            gripper_action (float): Action to take for the gripper (close-open:[0, 0.08]).
        """

    @abstractmethod
    def move_robot_to_joint_state(self, joint_state: np.ndarray, time_to_go: float = 4):
        """
        Move robot to the target joint state. Waypoint interpolation should be implemented.

        Args:
            joint_state (np.ndarray): Target joint state.
            time_to_go (float): Time to execute the command (in seconds).
        """

    @abstractmethod
    def get_multi_cam_pcd(self):
        """
        Get the combined point cloud from multiple cameras. Processing depth and RGB at the same time. RGB info gives better visualization for debugging supports but will slow down the process.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Points and colors of the combined point cloud. Both with shape (N, 3)
        """

    def get_joint_limits(self):
        """
        Get the joint limits of the robot.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Lower and upper joint limits.
        """
        return FRANKA_LOWER_LIMITS, FRANKA_UPPER_LIMITS

    def get_success(self, goal_angles, joint_angles=None, check_not_in_collision=False):
        """
        Compute success for env.
        Args:
            goal_angles (np.ndarray): (7,), unnormalized joint angles that are the goal.
            check_not_in_collision (bool): If True, also check that the robot is in a collision free state.
        Returns:
            success (bool): True if the robot is at the goal configuration and not in collision.
        """
        if joint_angles is None:
            current_ee = self.get_ee_pose()
        else:
            current_ee = self.get_ee_pose_from_joint_angles(joint_angles)
        goal_ee = self.get_ee_pose_from_joint_angles(goal_angles)
        position_error = 100 * np.linalg.norm(current_ee[:3] - goal_ee[:3])  # in cm
        current_ee_quat = Quaternion(current_ee[3:])
        goal_ee_quat = Quaternion(goal_ee[3:])
        orientation_error = (
            pyquaternion.Quaternion.absolute_distance(current_ee_quat, goal_ee_quat) * 180 / np.pi
        )
        success = (
            position_error < 1 and orientation_error < 15
        )  # 1cm position error, 15 degree orientation error
        return success, position_error, orientation_error

    def execute_joint_action(
        self,
        action_target: np.ndarray,
        start_angles: np.ndarray = None,
        gripper_width: float = None,
        speed: float = 0.1,
    ):
        """
        Execute joint control action on the robot with the given speed. Be careful with the manually set start_angles.

        Args:
            action_target (np.ndarray): Target joint angles.
            start_angles (np.ndarray, optional): Start joint angles.
            gripper_width (float, optional): Opening width of the gripper in meters.
            speed (float, optional): Speed for execution (rad/s).
        """
        ctrl_hz = self.ctrl_hz  # in Manimo the default is 60Hz
        if start_angles is None:
            start_angles = self.get_joint_angles()

        max_action = max(abs(action_target - start_angles))
        num_steps = max(int(max_action / speed * ctrl_hz), 10)

        rate = Rate(ctrl_hz)

        for step in range(num_steps):
            action_processed = (
                start_angles + (action_target - start_angles) * (step + 1) / num_steps
            )
            action_processed = np.clip(action_processed, FRANKA_LOWER_LIMITS, FRANKA_UPPER_LIMITS)
            self.step(joint_action=action_processed, gripper_action=gripper_width)
            rate.sleep()

    def execute_plan(
        self,
        plan,
        init_joint_angles,
        speed,
        proprio_feedback=False,
        render=False,
    ):
        """
        Execute a planned trajectory.

        Args:
            plan (list): List of joint angles along the path.
            init_joint_angles (np.ndarray): Initial joint angles of the robot.
            speed (float): Speed for execution (rad/s).
            soft_control (bool): Whether to use a controller with lower Kp, Kd gains. (Precision suffers, but could be useful for debugging / safety concerns)
            proprio_feedback (bool): Whether to use proprioceptive feedback.
            render (bool): Whether to render each execution step.

        Returns:
            Tuple[bool, float, list]: Success flag, joint error, and frames.
        """
        input("press Enter to move robot to the start config...")
        print("executing")
        self.execute_joint_action(action_target=init_joint_angles)

        input("press Enter to execute trajectory...")
        print("Executing plan...")
        print(f"Plan length: {len(plan)}")
        # take path and execute
        frames = []
        for plan_idx, state in enumerate(plan):
            if (plan_idx == 0) or proprio_feedback:
                start_angles = self.get_joint_angles()
            else:
                start_angles = plan[plan_idx - 1]

            self.execute_joint_action(
                action_target=state,
                start_angles=start_angles,
                speed=speed,
            )

            if render:
                im = self.render(mode="rgb_array")
                frames.append(im)

        achieved_joint_angles = self.get_joint_angles()
        joint_error = np.linalg.norm(achieved_joint_angles - plan[-1])
        success, position_error, orientation_error = self.get_success(
            plan[-1], check_not_in_collision=True
        )
        print(
            f"execution results:\njoint_err: {joint_error} deg\npos_err: {position_error} cm\nori_err: {orientation_error} deg"
        )
        return success, joint_error, frames

    def exclude_robot_pcd(self, points, colors=None, thred=0.01):
        """
        Exclude points belonging to the robot from the point cloud.

        Args:
            points (np.ndarray): Point cloud points.
            colors (np.ndarray, optional): Point cloud colors.
            thred (float): Threshold for excluding points.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Filtered points and colors.
        """
        config = self.get_joint_angles()
        centers, radii = self.collision_checker.spheres_cr(config)
        points = np.expand_dims(points, axis=2)

        centers = np.repeat(centers, points.shape[0], axis=0)
        sdf = np.linalg.norm((points - centers), axis=1) - radii
        is_robot_pcd = np.sum(sdf < thred, axis=1)
        scene_pcd_idx = np.where(is_robot_pcd == 0)[0]
        if colors is None:
            return points[scene_pcd_idx, :, 0]
        else:
            return points[scene_pcd_idx, :, 0], colors[scene_pcd_idx, :]

    def get_scene_pcd(self, debug=False):
        """
        Get the scene point cloud.

        Args:
            debug (bool): Whether to visualize the point cloud for debugging.

        Returns:
            np.ndarray: Scene point cloud.
        """
        (
            combined_pcd,
            combined_rgb,
        ) = self.get_multi_cam_pcd()  # get xyz and rgb info from multi-camera point cloud
        masked_pcd, masked_rgb = self.exclude_robot_pcd(combined_pcd, combined_rgb)

        if debug:
            save_pointcloud(
                "neural_mp/outputs/debug/debug_pcd.ply",
                np.array(masked_pcd),
                np.array(masked_rgb)[:, [2, 1, 0]],
            )
            self.visualize_ply("neural_mp/outputs/debug/debug_pcd.ply")

        return masked_pcd, masked_rgb

    @staticmethod
    def visualize_ply(ply_file_path):
        """
        Visualize a PLY file.

        Args:
            ply_file_path (str): Path to the PLY file.
        """
        point_cloud = o3d.io.read_point_cloud(ply_file_path)
        o3d.visualization.draw_geometries([point_cloud])

    def fk_batched(self, configs: torch.Tensor):
        """
        Perform forward kinematics on batched configurations.

        Args:
            configs (torch.Tensor): Batched joint configurations.

        Returns:
            torch.Tensor: Transformations for the end effector.
        """
        B = configs.shape[0]
        C = configs.shape[1]

        gripper_pos = self.get_gripper_width()
        if C == 7:
            configs = torch.cat(
                [configs, torch.ones([B, 1]).cuda() * gripper_pos / 2],
                axis=1,
            )
        link_transforms = self.collision_checker.compute_transformations(configs)
        return link_transforms[:, 8]

    def transform_in_hand_obj_batched(self, joint_angles, trans_in_hand2eef):
        """
        Transform an in-hand object given batched joint angles.

        Args:
            joint_angles (np.ndarray): Batched joint angles.
            trans_in_hand2eef (torch.Tensor): Transformation from in-hand object to end effector.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Global positions, orientations, and transformations.
        """

        def rotation_matrix_to_quaternion(matrix):
            m = matrix
            qw = torch.sqrt(1 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]) / 2
            qx = (m[:, 2, 1] - m[:, 1, 2]) / (4 * qw)
            qy = (m[:, 0, 2] - m[:, 2, 0]) / (4 * qw)
            qz = (m[:, 1, 0] - m[:, 0, 1]) / (4 * qw)
            return torch.cat(
                [qw.unsqueeze(1), qx.unsqueeze(1), qy.unsqueeze(1), qz.unsqueeze(1)], dim=1
            )

        panda_hand = self.fk_batched(joint_angles)  # Bx4x4

        final = panda_hand @ trans_in_hand2eef
        self.final_trans = final.cpu().numpy()
        global_pos = final[:, :3, 3]
        global_ori = rotation_matrix_to_quaternion(final[:, :3, :3])
        final = final
        return global_pos, global_ori, final

    def compute_in_hand_pcd(self, joint_angles, num_points, in_hand_params):
        """
        Compute the in-hand point cloud.

        Args:
            joint_angles (np.ndarray): Joint angles.
            num_points (int): Number of points in the point cloud.
            in_hand_params (np.ndarray): In-hand object parameters (type, size, position, orientation).

        Returns:
            torch.Tensor: In-hand point cloud.
        """
        in_hand_type = ["box", "cylinder", "sphere"][int(in_hand_params[0])]
        in_hand_size = in_hand_params[1:4]
        in_hand_pos = in_hand_params[4:7]
        in_hand_ori = in_hand_params[7:11]

        r = R_scipy.from_quat(in_hand_ori)
        in_hand_rotation_matrix = r.as_matrix()

        trans_in_hand2eef = torch.eye(4, device=torch.device("cuda"))
        trans_in_hand2eef[:3, 3] = torch.from_numpy(in_hand_pos).cuda()
        trans_in_hand2eef[:3, :3] = torch.from_numpy(in_hand_rotation_matrix).cuda()

        joint_angles_batched = torch.from_numpy(joint_angles).cuda()  # Bx7

        global_pos, global_ori, _ = self.transform_in_hand_obj_batched(
            joint_angles_batched, trans_in_hand2eef
        )

        if in_hand_type == "box":
            cuboid_centers = global_pos.float()
            cuboid_quaternions = global_ori.float()
            cuboid_dims = (
                torch.from_numpy(in_hand_size)
                .cuda()
                .unsqueeze(0)
                .repeat(joint_angles.shape[0], 1)
                .float()
            )
            in_hand_obj = TorchCuboids(
                cuboid_centers.unsqueeze(1),
                cuboid_dims.unsqueeze(1),
                cuboid_quaternions.unsqueeze(1),
            )
        elif in_hand_type == "cylinder":
            cylinder_centers = global_pos.float()
            cylinder_quaternions = global_ori.float()
            cylinder_dims = (
                torch.from_numpy(in_hand_size)
                .cuda()
                .unsqueeze(0)
                .repeat(joint_angles.shape[0], 1)
                .float()
            )
            in_hand_obj = TorchCylinders(
                cylinder_centers.unsqueeze(1), cylinder_dims, cylinder_quaternions.unsqueeze(1)
            )
        elif in_hand_type == "sphere":
            sphere_centers = global_pos.float()
            sphere_dims = (
                torch.from_numpy(in_hand_size)
                .cuda()
                .unsqueeze(0)
                .repeat(joint_angles.shape[0], 1)
                .float()
            )
            in_hand_obj = TorchSpheres(sphere_centers.unsqueeze(1), sphere_dims)
        in_hand_pcd = in_hand_obj.sample_surface(num_points)[:, 0]
        return in_hand_pcd

    def add_in_hand_pcd(
        self,
        pcd: torch.Tensor,
        joint_angles: np.ndarray,
        in_hand_params: np.ndarray,
        num_in_hand_points=500,
    ):
        """
        Add the in-hand point cloud to the pre-computed point cloud.

        Args:
            pcd (torch.Tensor): Pre-computed point cloud.
            joint_angles (np.ndarray): Joint angles.
            in_hand_params (np.ndarray): In-hand object parameters (type, size, position, orientation) 1+3+3+4. type{0: box, 1: cylinder, 2: sphere}.
            num_in_hand_points (int): Number of in-hand points.

        Returns:
            torch.Tensor: Combined point cloud.
        """
        # if type is not specified, default to use box
        if len(in_hand_params) == 10:
            in_hand_params = np.append(0, in_hand_params)
        pcd_size = pcd.shape[1]
        in_hand_pcd = self.compute_in_hand_pcd(joint_angles, num_in_hand_points, in_hand_params)
        combined_pcd = torch.cat((pcd, in_hand_pcd), dim=1)
        final_pcd = vectorized_subsample(combined_pcd, dim=1, num_points=pcd_size)
        return final_pcd


class FrankaRealEnvManimo(FrankaRealEnv):
    def __init__(
        self,
        camera_width=640,
        camera_height=480,
        cam_only=False,
        arm_only=False,
        depth_only=False,
        no_cam=False,
    ):
        """
        Initialize the environment.

        Args:
            camera_width (int): Width of the camera observation.
            camera_height (int): Height of the camera observation.
            cam_only (bool): Whether to launch the environment with camera only.
            arm_only (bool): Whether to launch the environment with franka arm only.
            depth_only (bool): Whether to launch the cameras with depth only.
            no_cam (bool): Whether to launch the environment without camera.
        """
        self.setup_robot(
            camera_width,
            camera_height,
            cam_only,
            arm_only,
            depth_only,
            no_cam,
        )
        self.collision_checker = FrankaCollisionChecker()
        self.setup_configs()

    def setup_robot(
        self,
        camera_width=640,
        camera_height=480,
        cam_only=False,
        arm_only=False,
        depth_only=False,
        no_cam=False,
    ):
        """
        Instantiate the robot in the real world

        Args:
            camera_width (int): Width of the camera observation.
            camera_height (int): Height of the camera observation.
            cam_only (bool): Whether to launch the environment with camera only.
            arm_only (bool): Whether to launch the environment with franka arm only.
            depth_only (bool): Whether to launch the cameras with depth only.
            no_cam (bool): Whether to launch the environment without camera.
        """
        hydra.initialize(config_path="../../manimo/manimo/conf/", job_name="collect_demos_test")
        env_cfg = hydra.compose(config_name="env")
        if cam_only:
            actuators_cfg = []
        elif arm_only:
            actuators_cfg = hydra.compose(config_name="actuators_arm")
        else:
            actuators_cfg = hydra.compose(config_name="actuators")

        self.hom = {}
        if no_cam:
            sensors_cfg = []
        else:
            if depth_only:
                sensors_cfg = hydra.compose(config_name="sensors_depth")
            else:
                sensors_cfg = hydra.compose(config_name="sensors")

            for cam_key in sensors_cfg["camera"]:
                sensors_cfg["camera"][cam_key]["camera_cfg"]["img_width"] = camera_width
                sensors_cfg["camera"][cam_key]["camera_cfg"]["img_height"] = camera_height
                cam_idx = int(cam_key[-1])
                self.hom[cam_key] = HomographyTransform(
                    f"img{cam_idx}",
                    transform_file="hom",
                    cam_cfg=sensors_cfg["camera"][cam_key]["camera_cfg"],
                )

        faulthandler.enable()
        self.env = SingleArmEnv(sensors_cfg, actuators_cfg, env_cfg)  # no reset at the low level
        self.im_width = camera_width
        self.im_height = camera_height
        self.ctrl_hz = actuators_cfg.arm.arm0.arm_cfg.hz if not cam_only else None
        if not cam_only:
            self.canonical_joint_pose = np.array(actuators_cfg.arm.arm0.arm_cfg.home)

        if not cam_only:
            assert type(self.env.actuators[0]) == FrankaArm
            self.robot_model = self.env.actuators[0].robot.robot_model
        self.fk_sampler = FrankaSampler("cpu", use_cache=False)
        self.urdf = urchin.URDF.load(FrankaRobot.urdf)

    def robot_reconnect(self, arm_only=False):
        """
        Reconnect the robot arm.

        Args:
            arm_only (bool): Whether to reconnect only the arm.
        """
        if arm_only:
            actuators_cfg = hydra.compose(config_name="actuators_arm")
        else:
            actuators_cfg = hydra.compose(config_name="actuators")
        self.env.actuators = [
            hydra.utils.instantiate(actuators_cfg[actuator_type][actuator])
            for actuator_type in actuators_cfg
            for actuator in actuators_cfg[actuator_type]
        ]

    def get_joint_angles(self):
        """
        Get the joint angles of the robot.

        Returns:
            np.ndarray: 7-dof joint angles.
        """
        obs = self.env.get_actuator_obs()
        return obs["q_pos"]

    def get_joint_vels(self):
        """
        Get the joint velocities of the robot.
        Returns:
            np.ndarray: 7-dof joint velocities.
        """
        obs = self.env.get_obs()
        return obs["q_vel"]

    def get_ee_pose(self):
        """
        Get the end effector pose.

        Returns:
            np.ndarray: 7D end effector pose (x, y, z, xyzw).
        """
        obs = self.env.get_obs()
        return np.concatenate([obs["eef_pos"], obs["eef_rot"]])

    def get_gripper_width(self):
        """
        Get the gripper width.

        Returns:
            float: Gripper width.
        """
        obs = self.env.get_obs()
        return obs["eef_gripper_width"]

    def compute_ik(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        q0: np.ndarray,
        tol: float = 1e-3,
    ) -> Tuple[np.ndarray, bool]:
        """
        Compute inverse kinematics given desired EE pose.

        Args:
            position (np.ndarray): 3D position of the end effector.
            orientation (np.ndarray): Orientation of the end effector.
            q0 (np.ndarray): Initial solution for IK (current joint positions).
            tol (float): Tolerance for the IK solution.

        Returns:
            Tuple[np.ndarray, bool]: Joint positions and success flag.
        """

        position_t = torch.Tensor(position)
        orientation_t = torch.Tensor(orientation)
        q0_t = torch.Tensor(q0)

        # Call IK
        joint_pos_output = self.robot_model.inverse_kinematics(
            position_t, orientation_t, rest_pose=q0_t
        )

        # Check result
        pos_output, quat_output = self.robot_model.forward_kinematics(joint_pos_output)
        pose_desired = T.from_rot_xyz(R.from_quat(orientation_t), position_t)
        pose_output = T.from_rot_xyz(R.from_quat(quat_output), pos_output)
        err = torch.linalg.norm((pose_desired * pose_output.inv()).as_twist())
        ik_sol_found = err < tol

        return joint_pos_output.numpy(), ik_sol_found.numpy()

    def get_joint_from_ee_pose(self, target_ee_pose):
        """
        Get the joint angles from the end effector pose using Inverse Kinematics.

        Args:
            target_ee_pose (np.ndarray): 7D end effector pose (x, y, z, xyzw).

        Returns:
            Tuple[np.ndarray, bool]: Joint angles and success flag.
        """
        joint_pos, success = self.compute_ik(
            target_ee_pose[:3], target_ee_pose[3:], self.get_joint_angles()
        )

        return joint_pos, success

    def get_ee_pose_from_joint_angles(self, joint_angles):
        """
        Get the end effector pose from the joint angles using Forward Kinematics.

        Args:
            joint_angles (np.ndarray): 7-dof joint angles.

        Returns:
            np.ndarray: 7D end effector pose.
        """
        joint_angles_t = torch.Tensor(joint_angles)
        pos_output, quat_output = self.robot_model.forward_kinematics(joint_angles_t)
        return np.concatenate([pos_output.numpy(), quat_output.numpy()])

    def reset(self):
        """
        Reset joint angles to canonical pose.
        """
        self.move_robot_to_joint_state(joint_state=self.canonical_joint_pose, time_to_go=4)

    def setup_configs(self, start_config=None, goal_angles=None):
        """
        Setup start and goal configurations.

        Args:
            start_config (np.ndarray, optional): Start joint configuration.
            goal_angles (np.ndarray, optional): Goal joint angles.
        """
        self.start_config = start_config
        self.goal_angles = goal_angles
        self.goal_ee = (
            self.get_ee_pose_from_joint_angles(self.goal_angles)
            if goal_angles is not None
            else None
        )

    def step(self, joint_action: np.ndarray = None, gripper_action: float = None):
        """
        Step the robot to the target joint state without any interpolation.
        Be careful about your step size, large joint displacement might hurt the robot motor.

        Args:
            joint_action (np.ndarray): Action to take for the arm. (absolute joint angles)
            gripper_action (float): Action to take for the gripper (close-open:[0, 0.08]).
        """
        # if no action is given, print a warning
        if joint_action is None and gripper_action is None:
            print("Warning: no action is given for step function")
        # if both actions are given
        elif joint_action is not None and gripper_action is not None:
            self.env.step([joint_action, gripper_action])
        # if only joint action is given
        elif joint_action is not None and gripper_action is None:
            assert type(self.env.actuators[0]) == FrankaArm
            frankarm = self.env.actuators[0]
            frankarm.step(joint_action)
        # if only gripper action is given
        elif joint_action is None and gripper_action is not None:
            assert type(self.env.actuators[1]) == PolymetisGripper
            gripper = self.env.actuators[1]
            gripper.step(gripper_action)

    def get_observation(self, cam_idx=1):
        """
        Get the observation from the environment.

        Args:
            cam_idx (int): Camera index.

        Returns:
            OrderedDict: Observation dictionary.
        """
        obs = self.env.get_obs()
        ret = OrderedDict()

        # "object" key contains object information
        ret["eef_pos"] = obs["eef_pos"].astype(np.float32)
        ret["eef_quat"] = obs["eef_rot"].astype(np.float32)
        ret["gripper_width"] = np.array([obs["eef_gripper_width"]]).astype(np.float32)
        ret[f"cam{cam_idx}_rgb"] = obs[f"cam{cam_idx}"][0].astype(np.float32)
        ret[f"cam{cam_idx}_depth"] = obs[f"cam{cam_idx}_depth"][0].astype(np.float32)

        return ret

    def render(self, mode="rgb_array", height=224, width=224, camera_name="cam1"):
        """
        Render from real world camera to either an on-screen window or off-screen to RGB array.

        Args:
            mode (str): "human" for on-screen rendering or "rgb_array" for off-screen rendering.
            height (int): Height of image to render (only used if mode is "rgb_array").
            width (int): Width of image to render (only used if mode is "rgb_array").
            camera_name (str): Camera name to use for rendering.

        Returns:
            np.ndarray: Rendered image.
        """
        if mode == "human":
            raise NotImplementedError("mode=human is not implemented".format(mode))
        elif mode == "rgb_array":
            return cv2.resize(self.env.get_obs()[f"{camera_name}"][0], (height, width))
        elif mode == "depth":
            return cv2.resize(self.env.get_obs()[f"{camera_name}_depth"][0], (height, width))
        elif mode == "rgb_cropped":
            return crop_and_resize_back_view(
                self.env.get_obs()[f"{camera_name}"][0], (height, width)
            )
        else:
            raise NotImplementedError("mode={} is not implemented".format(mode))

    def close(self):
        """
        Close the environment.
        """
        print("Closing...")
        self.env.close()
        time.sleep(1)
        print("Environment Closed")

    def move_robot_to_joint_state(self, joint_state: np.ndarray, time_to_go: float = 4):
        """
        Move robot to the target joint state using a controller with lower Kp and Kd gains.
        Waypoint interpolation will be done at the low level.

        Args:
            joint_state (np.ndarray): Target joint state.
            time_to_go (float): Time to execute the command (in seconds).
        """

        if joint_state is not None:
            assert (
                type(self.env.actuators[0]) == FrankaArm
            )  # should always make sure you are calling the right class
            frankarm = self.env.actuators[0]
            frankarm.soft_ctrl(joint_state, time_to_go)

    def move_to_joint_pos_from_ee_pose(self, target_ee_pose, time_to_go=4):
        """
        Set the joint angles from the end effector pose.

        Args:
            target_ee_pose (np.ndarray): 7D end effector pose (x, y, z, xyzw).
            time_to_go (float): Time to execute the command (in seconds).

        Returns:
            float: Error in achieving the target end effector pose.
        """
        # assert target pose is not current pose
        joint_pos, success = self.get_joint_from_ee_pose(target_ee_pose)

        if success:
            self.move_robot_to_joint_state(joint_pos, time_to_go)
        else:
            print(
                "Warning: Unable to find valid joint target. Skipping set_joint_pos_from_ee_pose command..."
            )

        achieved_ee_pose = self.get_ee_pose()
        error = np.linalg.norm(achieved_ee_pose - target_ee_pose)
        return error

    def smooth(self, path, timesteps, obstacle_points):
        """
        Smooth a given path using cubic smoothing.

        Args:
            path (list): List of joint configurations.
            timesteps (int): Number of timesteps for the smoothed path.
            obstacle_points (np.ndarray): Points representing obstacles.

        Returns:
            list: Smoothed path.
        """
        curve = smooth_cubic(
            path,
            lambda q: self.collision_checker.check_collision(q, obstacle_points),
            np.radians(0.1) * np.ones(7),
            FRANKA_VELOCITY_LIMIT,
            FRANKA_ACCELERATION_LIMIT,
        )
        ts = (curve.x[-1] - curve.x[0]) / (timesteps - 1)
        return [curve(ts * i) for i in range(timesteps)]

    def mp_to_joint_target(
        self,
        start_angles,
        target_angles,
        obstacle_points,
        planning_time=1,
        num_waypoints=50,
        state_validity_checking_resolution=0.0001,
    ):
        """
        Plan a path from the current joint state to the target joint state.

        Args:
            start_angles (np.ndarray): Start joint angles. (1, 7)
            target_angles (np.ndarray): Target joint angles. (1, 7)
            obstacle_points (np.ndarray): Points representing obstacles. (n, 3)
            planning_time (float): Time allocated for planning.
            num_waypoints (int): Number of waypoints in the final trajectory.
            state_validity_checking_resolution (float): Resolution for state validity checking.

        Returns:
            list: Planned path as a list of joint angles.
        """
        # ompl uses doubles
        start_angles = start_angles.astype(np.float64)
        target_angles = target_angles.astype(np.float64)
        lower_limits, upper_limits = self.get_joint_limits()
        lower_limits = lower_limits.astype(np.float64)
        upper_limits = upper_limits.astype(np.float64)

        def isStateValid(state):
            joint_pos = np.zeros(7)
            for i in range(7):
                joint_pos[i] = state[i]
            valid = not self.collision_checker.check_collision(joint_pos, obstacle_points)
            return valid

        # set up planning space for ompl
        space = ob.RealVectorStateSpace(7)
        bounds = ob.RealVectorBounds(7)
        for i in range(7):
            bounds.setLow(i, lower_limits[i])
            bounds.setHigh(i, upper_limits[i])
        space.setBounds(bounds)
        si = ob.SpaceInformation(space)
        si.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))
        si.setStateValidityCheckingResolution(state_validity_checking_resolution)

        # setup start state
        start = ob.State(space)
        for i in range(7):
            start()[i] = start_angles[i]
        start_valid = isStateValid(start)
        if not start_valid:
            print("Start state is not valid!")
            return None

        # setup goal state
        goal = ob.State(space)
        for i in range(7):
            goal()[i] = target_angles[i]

        goal_valid = isStateValid(goal)
        print(f"Goal valid: {goal_valid}")

        # create a problem instance
        pdef = ob.ProblemDefinition(si)
        # set the start and goal states
        pdef.setStartAndGoalStates(start, goal)
        # create a planner for the defined space
        pdef.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))
        planner = og.AITstar(si)
        # set the problem we are trying to solve for the planner
        planner.setProblemDefinition(pdef)
        # perform setup steps for the planner
        planner.setup()
        solved = planner.solve(planning_time)
        converted_path = []
        if solved:
            path = pdef.getSolutionPath()
            path = path.getStates()
            for i in range(len(path)):
                converted_path.append(np.array([path[i][j] for j in range(7)]))
            converted_path = np.asarray(self.smooth(converted_path, num_waypoints, obstacle_points))
            # check if all components of the path are within the joint limits:
            lower_limits, upper_limits = self.get_joint_limits()
            for state in converted_path:
                if np.any(state < lower_limits) or np.any(state > upper_limits):
                    print("Path is out of joint limits!")
                    return None, [], []

            return converted_path
        else:
            return None, [], []

    def get_fixed_depth(
        self,
        cam_name="cam1",
    ):
        """
        Get the depth image from the environment.

        Args:
            cam_name (str): Camera name.

        Returns:
            np.ndarray: Depth image.
        """
        return self.env.get_obs()[f"{cam_name}_depth"][0]

    def get_fixed_rgb(
        self,
        cam_name="cam1",
    ):
        """
        Get the RGB image from the environment.

        Args:
            cam_name (str): Camera name.

        Returns:
            np.ndarray: RGB image.
        """
        return self.env.get_obs()[f"{cam_name}"][0]

    def get_raw_pcd_single_cam(self, cam_idx=1, filter=True, denoise=False, debug=False):
        """
        Get the raw point cloud from a single camera.

        Args:
            cam_idx (int): Camera index.
            filter (bool): Whether to filter the point cloud.
            denoise (bool): Whether to denoise the point cloud.
            debug (bool): Whether to save and visualize the point cloud for debugging.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Points and colors of the point cloud.
        """
        img_numpy = self.get_fixed_rgb(f"cam{cam_idx}")
        depth_numpy = self.get_fixed_depth(f"cam{cam_idx}")

        output_dir = "neural_mp/outputs"
        raw_pc = self.hom[f"cam{cam_idx}"].get_pointcloud(depth_numpy).reshape(-1, 3)
        raw_img = img_numpy[:, :, ::-1].reshape(-1, 3) / 255.0
        if debug:
            save_pointcloud(
                output_dir + "/debug/raw_pc.ply",
                np.array(raw_pc),
                np.array(raw_img)[:, [2, 1, 0]],
            )
            self.visualize_ply(output_dir + "/debug/raw_pc.ply")

        if filter:
            points, colors = self.hom[f"cam{cam_idx}"].get_filtered_pc(
                raw_pc.reshape(-1, 3), raw_img.reshape(-1, 3)
            )
            if denoise:
                points, colors = self.hom[f"cam{cam_idx}"].denoise_pc(points, colors)
            if debug:
                save_pointcloud(
                    output_dir + "/debug/filtered_pc.ply",
                    np.array(points),
                    np.array(colors)[:, [2, 1, 0]],
                )
                self.visualize_ply(output_dir + "/debug/filtered_pc.ply")
            return points, colors

        return raw_pc, raw_img

    def get_multi_cam_pcd(
        self,
        debug_raw_pcd=False,
        filter=True,
        denoise=False,
    ):
        """
        Get the combined point cloud from multiple cameras. Processing depth and RGB at the same time. RGB info gives better visualization for debugging supports but will slow down the process.

        Args:
            debug_raw_pcd (bool): Whether to debug the raw point cloud.
            filter (bool): Whether to filter the point cloud.
            denoise (bool): Whether to denoise the point cloud.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Points and colors of the combined point cloud.
        """
        combined_pcd = None
        combined_rgb = None
        for cam_key in self.hom.keys():
            cam_idx = int(cam_key[-1])
            pcd, rgb = self.get_raw_pcd_single_cam(
                cam_idx=cam_idx,
                filter=filter,
                denoise=denoise,
                debug=debug_raw_pcd,
            )
            if combined_pcd is None:
                combined_pcd = pcd
                combined_rgb = rgb
            else:
                combined_pcd = np.concatenate((combined_pcd, pcd), axis=0)
                combined_rgb = np.concatenate((combined_rgb, rgb), axis=0)

        return combined_pcd, combined_rgb

    def get_scene_pcd(
        self,
        debug_raw_pcd=False,
        debug_combined_pcd=False,
        save_pcd=False,
        save_file_name="combined",
        filter=True,
        denoise=False,
    ):
        """
        Get the segmented scene point cloud from multiple cameras.

        Args:
            debug_raw_pcd (bool): Whether to debug the raw point cloud.
            debug_combined_pcd (bool): Whether to debug the combined point cloud.
            save_pcd (bool): Whether to save the combined point cloud.
            save_file_name (str): File name for saving the combined point cloud.
            filter (bool): Whether to filter the point cloud.
            denoise (bool): Whether to denoise the point cloud.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Points and colors of the combined point cloud.
        """
        combined_pcd, combined_rgb = self.get_multi_cam_pcd(
            debug_raw_pcd=debug_raw_pcd,
            filter=filter,
            denoise=denoise,
        )

        scene_pcd, scene_rgb = self.exclude_robot_pcd(combined_pcd, combined_rgb)

        if debug_combined_pcd:
            save_pointcloud(
                "neural_mp/outputs/debug/combined_pcd.ply",
                np.array(scene_pcd),
                np.array(scene_rgb)[:, [2, 1, 0]],
            )
            self.visualize_ply("neural_mp/outputs/debug/combined_pcd.ply")

        if save_pcd:
            current_time = datetime.datetime.now()
            np.save(
                f"real_world_test_set/collected_pcds/{save_file_name}_pcd_{current_time}.npy",
                scene_pcd,
            )
            np.save(
                f"real_world_test_set/collected_pcds/{save_file_name}_rgb_{current_time}.npy",
                scene_rgb,
            )

        return scene_pcd, scene_rgb

    def get_multi_cam_pcd_fast(
        self,
        debug_raw_pcd=False,
        debug_combined_pcd=False,
        save_pcd=False,
        filter=True,
        denoise=False,
        downsample=10000,
    ):
        """
        Get the combined point cloud from multiple cameras quickly by only processing depth information.

        Args:
            debug_raw_pcd (bool): Whether to debug the raw point cloud.
            debug_combined_pcd (bool): Whether to debug the combined point cloud.
            save_pcd (bool): Whether to save the combined point cloud.
            filter (bool): Whether to filter the point cloud.
            denoise (bool): Whether to denoise the point cloud.
            downsample (int): Number of points to downsample.

        Returns:
            np.ndarray: Masked point cloud.
        """
        obs = self.env.get_img_obs()
        depth1 = obs["cam1_depth"][0]
        depth2 = obs["cam2_depth"][0]
        depth3 = obs["cam3_depth"][0]
        depth4 = obs["cam4_depth"][0]

        raw_pcd1 = self.hom["cam1"].get_pointcloud(depth1).reshape(-1, 3)
        raw_pcd2 = self.hom["cam2"].get_pointcloud(depth2).reshape(-1, 3)
        raw_pcd3 = self.hom["cam3"].get_pointcloud(depth3).reshape(-1, 3)
        raw_pcd4 = self.hom["cam4"].get_pointcloud(depth4).reshape(-1, 3)

        raw_combined_pcd = np.concatenate((raw_pcd1, raw_pcd2, raw_pcd3, raw_pcd4), axis=0)

        random_obstacle_indices = np.random.choice(
            len(raw_combined_pcd), size=downsample, replace=False
        )
        downsampled_pcd = raw_combined_pcd[random_obstacle_indices]

        filtered_pcd = self.hom["cam1"].get_filtered_pc(raw_combined_pcd)

        masked_pcd = self.exclude_robot_pcd(filtered_pcd)
        if debug_combined_pcd:
            masked_rgb = np.zeros((len(masked_pcd), 3))
            save_pointcloud(
                "neural_mp/outputs/debug/combined_pcd.ply",
                np.array(masked_pcd),
                np.array(masked_rgb),
            )
            self.visualize_ply("neural_mp/outputs/debug/combined_pcd.ply")

        return masked_pcd
