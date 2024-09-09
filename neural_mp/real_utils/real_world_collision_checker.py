"""
Using Spherical representation of the Franka Robot for real world collision checking
Model comes from STORM: https://github.com/NVlabs/storm/blob/e53556b64ca532e836f6bfd50893967f8224980e/content/configs/robot/franka_real_robot.yml
"""

import logging
import time

import kornia.geometry.conversions as conversions
import numpy as np
import open3d as o3d
import torch
from geometrout.primitive import Sphere
from robofin.pointcloud.numpy import transform_pointcloud
from robofin.robots import FrankaRobot
from urchin import URDF

from neural_mp.real_utils.homography_utils import save_pointcloud
from neural_mp.utils.geometry import TorchCuboids, TorchSpheres

SELF_COLLISION_SPHERES = [
    ("panda_link0", [-0.08, 0.0, 0.05], 0.06),
    ("panda_link0", [-0.0, 0.0, 0.05], 0.08),
    ("panda_link1", [0.0, -0.08, 0.0], 0.1),
    ("panda_link1", [0.0, -0.03, 0.0], 0.1),
    ("panda_link1", [0.0, 0.0, -0.12], 0.06),
    ("panda_link1", [0.0, 0.0, -0.17], 0.06),
    ("panda_link2", [0.0, 0.0, 0.03], 0.1),
    ("panda_link2", [0.0, 0.0, 0.08], 0.1),
    ("panda_link2", [0.0, -0.12, 0.0], 0.06),
    ("panda_link2", [0.0, -0.17, 0.0], 0.06),
    ("panda_link3", [0.0, 0.0, -0.06], 0.05),
    ("panda_link3", [0.0, 0.0, -0.1], 0.06),
    ("panda_link3", [0.08, 0.06, 0.0], 0.055),
    ("panda_link3", [0.08, 0.02, 0.0], 0.055),
    ("panda_link4", [0.0, 0.0, 0.02], 0.055),
    ("panda_link4", [0.0, 0.0, 0.06], 0.055),
    ("panda_link4", [-0.08, 0.095, 0.0], 0.06),
    ("panda_link4", [-0.08, 0.06, 0.0], 0.055),
    ("panda_link5", [0.0, 0.055, 0.0], 0.05),
    ("panda_link5", [0.0, 0.085, 0.0], 0.055),
    ("panda_link5", [0.0, 0.0, -0.22], 0.05),
    ("panda_link5", [0.0, 0.05, -0.18], 0.045),
    ("panda_link5", [0.015, 0.08, -0.14], 0.03),
    ("panda_link5", [0.015, 0.085, -0.11], 0.03),
    ("panda_link5", [0.015, 0.09, -0.08], 0.03),
    ("panda_link5", [0.015, 0.095, -0.05], 0.03),
    ("panda_link5", [-0.015, 0.08, -0.14], 0.03),
    ("panda_link5", [-0.015, 0.085, -0.11], 0.03),
    ("panda_link5", [-0.015, 0.09, -0.08], 0.03),
    ("panda_link5", [-0.015, 0.095, -0.05], 0.03),
    ("panda_link6", [0.0, 0.0, 0.0], 0.05),
    ("panda_link6", [0.08, 0.035, 0.0], 0.052),
    ("panda_link6", [0.08, -0.01, 0.0], 0.05),
    ("panda_link7", [0.0, 0.0, 0.07], 0.05),
    ("panda_link7", [0.02, 0.04, 0.08], 0.025),
    ("panda_link7", [0.04, 0.02, 0.08], 0.025),
    ("panda_link7", [0.04, 0.06, 0.085], 0.02),
    ("panda_link7", [0.06, 0.04, 0.085], 0.02),
    ("panda_hand", [0.0, -0.08, 0.01], 0.03),
    ("panda_hand", [0.0, -0.045, 0.01], 0.03),
    ("panda_hand", [0.0, -0.015, 0.01], 0.03),
    ("panda_hand", [0.0, 0.015, 0.01], 0.03),
    ("panda_hand", [0.0, 0.045, 0.01], 0.03),
    ("panda_hand", [0.0, 0.08, 0.01], 0.03),
    ("panda_hand", [0.0, 0.065, -0.02], 0.04),
    ("panda_hand", [0.0, -0.08, 0.05], 0.05),
    ("panda_hand", [0.0, -0.045, 0.05], 0.05),
    ("panda_hand", [0.0, -0.015, 0.05], 0.05),
    ("panda_hand", [0.0, 0.015, 0.05], 0.05),
    ("panda_hand", [0.0, 0.045, 0.05], 0.05),
    ("panda_hand", [0.0, 0.08, 0.05], 0.05),
    ("panda_hand", [0.0, 0.08, 0.08], 0.05),
    ("panda_hand", [0.0, -0.08, 0.08], 0.05),
    ("panda_hand", [0.0, 0.05, 0.08], 0.05),
    ("panda_hand", [0.0, -0.05, 0.08], 0.05),
    ("panda_hand", [0.0, 0.0, 0.08], 0.05),
]

SELF_COLLISION_SPHERES_HALF = [
    ("panda_link5", [0.0, 0.055, 0.0], 0.05),
    ("panda_link5", [0.0, 0.085, 0.0], 0.055),
    ("panda_link6", [0.0, 0.0, 0.0], 0.05),
    ("panda_link6", [0.08, 0.035, 0.0], 0.052),
    ("panda_link6", [0.08, -0.01, 0.0], 0.05),
    ("panda_link7", [0.0, 0.0, 0.07], 0.05),
    ("panda_link7", [0.02, 0.04, 0.08], 0.025),
    ("panda_link7", [0.04, 0.02, 0.08], 0.025),
    ("panda_link7", [0.04, 0.06, 0.085], 0.02),
    ("panda_link7", [0.06, 0.04, 0.085], 0.02),
    ("panda_hand", [0.0, -0.08, 0.01], 0.03),
    ("panda_hand", [0.0, -0.045, 0.01], 0.03),
    ("panda_hand", [0.0, -0.015, 0.01], 0.03),
    ("panda_hand", [0.0, 0.015, 0.01], 0.03),
    ("panda_hand", [0.0, 0.045, 0.01], 0.03),
    ("panda_hand", [0.0, 0.08, 0.01], 0.03),
    ("panda_hand", [0.0, 0.065, -0.02], 0.04),
    ("panda_hand", [0.0, -0.08, 0.05], 0.05),
    ("panda_hand", [0.0, -0.045, 0.05], 0.05),
    ("panda_hand", [0.0, -0.015, 0.05], 0.05),
    ("panda_hand", [0.0, 0.015, 0.05], 0.05),
    ("panda_hand", [0.0, 0.045, 0.05], 0.05),
    ("panda_hand", [0.0, 0.08, 0.05], 0.05),
    ("panda_hand", [0.0, 0.08, 0.08], 0.05),
    ("panda_hand", [0.0, -0.08, 0.08], 0.05),
    ("panda_hand", [0.0, 0.05, 0.08], 0.05),
    ("panda_hand", [0.0, -0.05, 0.08], 0.05),
    ("panda_hand", [0.0, 0.0, 0.08], 0.05),
]

# additional parameters for collision checking cuboids
SIZE_C = [[0.26, 0.08, 0.2]]
CENTERS_C = [[0.0, 0.0, 0.15]]
ORI_C = [[1.0, 0.0, 0.0, 0.0]]  # quaternion (w, x, y, z)
# additional parameters for collision checking spheres
CENTERS_S = []
RAIUS_S = []
for item in SELF_COLLISION_SPHERES_HALF:
    CENTERS_S.append(item[1])
    RAIUS_S.append(item[2])


def visualize_ply(ply_file_path):
    """
    Visualize a PLY file.

    Args:
        ply_file_path (str): Path to the PLY file.
    """
    point_cloud = o3d.io.read_point_cloud(ply_file_path)
    o3d.visualization.draw_geometries([point_cloud])


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor):
    """
    Multiplies two batches of quaternions.
    q1: Tensor of shape (..., 4), representing the first batch of quaternions
    q2: Tensor of shape (..., 4), representing the second batch of quaternions
    Returns:
    q3: Tensor of shape (..., 4), representing the resulting batch of quaternions
    """
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)

    w3 = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x3 = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y3 = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z3 = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return torch.stack((w3, x3, y3, z3), dim=-1)


class FrankaCollisionChecker:
    def __init__(
        self,
        default_prismatic_value=0.025,
    ):
        """
        Initialize the FrankaCollisionChecker class.

        Args:
            default_prismatic_value (float): Default prismatic value for the gripper.
        """
        logging.getLogger("trimesh").setLevel("ERROR")

        self.default_prismatic_value = default_prismatic_value
        self.robot = URDF.load(FrankaRobot.urdf, lazy_load_meshes=True)
        # Set up the center points for calculating the FK position
        link_names = []
        centers = {}
        for s in SELF_COLLISION_SPHERES:
            if s[0] not in centers:
                link_names.append(s[0])
                centers[s[0]] = [s[1]]
            else:
                centers[s[0]].append(s[1])
        self.points = [(name, np.asarray(centers[name])) for name in link_names]

        self.collision_matrix = -np.inf * np.ones(
            (len(SELF_COLLISION_SPHERES), len(SELF_COLLISION_SPHERES))
        )

        link_ids = {link_name: idx for idx, link_name in enumerate(link_names)}
        # Set up the self collision distance matrix
        for idx1, (link_name1, center1, radius1) in enumerate(SELF_COLLISION_SPHERES):
            for idx2, (link_name2, center2, radius2) in enumerate(SELF_COLLISION_SPHERES):
                # Ignore all sphere pairs on the same link or adjacent links
                if abs(link_ids[link_name1] - link_ids[link_name2]) < 2:
                    continue
                self.collision_matrix[idx1, idx2] = radius1 + radius2

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dh_params = torch.tensor(
            [
                [0, 0, 0.333, 0],
                [-np.pi / 2, 0, 0, 0],
                [np.pi / 2, 0, 0.316, 0],
                [np.pi / 2, 0.0825, 0, 0],
                [-np.pi / 2, -0.0825, 0.384, 0],
                [np.pi / 2, 0, 0, 0],
                [np.pi / 2, 0.088, 0, 0],
                [0, 0, 0.107, 0],
                [0, 0, 0.0, -np.pi / 4],
            ]
        ).to(self.device)
        # initialize params for collision checking cuboids
        self.cuboids_size = SIZE_C
        self.cuboids_center = CENTERS_C
        self.cuboids_ori = ORI_C

    def set_cuboid_params(self, sizes, centers, oris):
        """
        Set the parameters for collision checking cuboids.

        Args:
            sizes (list): List of cuboid sizes.
            centers (list): List of cuboid centers.
            oris (list): List of cuboid orientations.
        """
        self.cuboids_size = sizes
        self.cuboids_center = centers
        self.cuboids_ori = oris

    def dh_transformation(self, alpha, a, d, q):
        """
        Compute the DH (Denavit-Hartenberg) transformation matrix.
        For more details about DH parameters, visit https://frankaemika.github.io/docs/control_parameters.html#denavithartenberg-parameters

        Args:
            alpha (float): DH parameter alpha.
            a (float): DH parameter a.
            d (float): DH parameter d.
            q (float): Joint angle.

        Returns:
            torch.Tensor: DH transformation matrix.
        """
        sin_q = torch.sin(q)
        cos_q = torch.cos(q)
        sin_a = torch.sin(alpha)
        cos_a = torch.cos(alpha)

        dh_transform = torch.zeros(q.shape[0], 4, 4).to(self.device)
        dh_transform[:, 0, 0] = cos_q
        dh_transform[:, 0, 1] = -sin_q
        dh_transform[:, 0, 2] = 0
        dh_transform[:, 0, 3] = a
        dh_transform[:, 1, 0] = sin_q * cos_a
        dh_transform[:, 1, 1] = cos_q * cos_a
        dh_transform[:, 1, 2] = -sin_a
        dh_transform[:, 1, 3] = -sin_a * d
        dh_transform[:, 2, 0] = sin_q * sin_a
        dh_transform[:, 2, 1] = cos_q * sin_a
        dh_transform[:, 2, 2] = cos_a
        dh_transform[:, 2, 3] = cos_a * d
        dh_transform[:, 3, 0] = 0
        dh_transform[:, 3, 1] = 0
        dh_transform[:, 3, 2] = 0
        dh_transform[:, 3, 3] = 1

        return dh_transform

    def compute_transformations(self, joint_configs):
        """
        Compute the transformations for all links given joint configurations.

        Args:
            joint_configs (torch.Tensor): Joint configurations.

        Returns:
            torch.Tensor: Link transformation matrices.
        """
        batch_size = joint_configs.shape[0]
        num_links = len(self.dh_params)
        link_transforms = (
            torch.eye(4).repeat(batch_size, num_links, 1, 1).to(self.device)
        )  # Initialize as identity matrices, link1-8 & hand

        for i in range(num_links):
            alpha = self.dh_params[i, 0]
            a = self.dh_params[i, 1]
            d = self.dh_params[i, 2]
            if i < 7:
                q = joint_configs[:, i] + self.dh_params[i, 3]
            else:
                q = self.dh_params[i, 3].unsqueeze(0).repeat(batch_size)
            dh_transform = self.dh_transformation(alpha, a, d, q)

            if i == 0:
                link_transforms[:, i, :, :] = dh_transform
            else:
                link_transforms[:, i, :, :] = torch.matmul(
                    link_transforms[:, i - 1, :, :], dh_transform
                )

        return link_transforms

    def spheres(self, config):
        """
        Get the spherical representation of the robot given a configuration.

        Args:
            config (np.ndarray): Joint configuration.

        Returns:
            list[Sphere]: List of Sphere objects.
        """
        cfg = np.ones(8)
        cfg[:7] = config
        cfg[-1] = self.default_prismatic_value
        fk = self.robot.link_fk(cfg, use_names=True)
        spheres = []
        for link_name, center, radius in SELF_COLLISION_SPHERES:
            spheres.append(Sphere((fk[link_name] @ np.array([*center, 1]))[:3], radius))
        return spheres

    def spheres_cr(self, config):
        """
        Get the centers and radii of the spheres representing the robot at a given configuration.

        Args:
            config (np.ndarray): Joint configuration.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Centers and radii of the spheres.
        """
        cfg = np.ones(8)
        cfg[:7] = config
        cfg[-1] = self.default_prismatic_value
        fk = self.robot.link_fk(cfg, use_names=True)
        centers = []
        radii = []
        for link_name, center, radius in SELF_COLLISION_SPHERES:
            centers.append((fk[link_name] @ np.array([*center, 1]))[:3])
            radii.append(radius)

        return np.asarray(centers).T.reshape((1, 3, len(centers))), np.asarray(radii).reshape(
            1, len(radii)
        )

    def torch_spheres(self, configs: torch.Tensor):
        """
        Get the spheres representing the robot's configuration in a batched manner.

        Args:
            configs (torch.Tensor): Joint configurations (B, C), where B is batch size, and C is 7 or 8.

        Returns:
            TorchSpheres: Spheres representing the robot's configuration.
        """
        assert configs.ndim == 2
        B = configs.shape[0]
        M = len(SELF_COLLISION_SPHERES_HALF)
        C = configs.shape[1]

        if C == 7:
            configs = torch.cat(
                [configs, torch.ones([B, 1], device=self.device) * self.default_prismatic_value],
                axis=1,
            )

        # right now not considering the finger dof
        link_transforms = self.compute_transformations(configs)
        fk_links = torch.cat(
            [
                link_transforms[:, 4:5].repeat(1, 2, 1, 1),
                link_transforms[:, 5:6].repeat(1, 3, 1, 1),
                link_transforms[:, 6:7].repeat(1, 5, 1, 1),
                link_transforms[:, 8:9].repeat(1, M - 10, 1, 1),
            ],
            dim=1,
        )

        center_offsets = (
            torch.cat([torch.Tensor(CENTERS_S), torch.ones(M, 1)], dim=1)
            .to(self.device)
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        centers = torch.matmul(fk_links, center_offsets)[:, :, :3, 0]
        radii = torch.Tensor(RAIUS_S).to(self.device).unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1)
        return TorchSpheres(
            centers.reshape(B, M, 3),
            radii.reshape(B, M, 1),
        )

    def torch_spheres_cuboids(self, configs: torch.Tensor):
        """
        Get the spheres and cuboids representing the robot's configuration in a batched manner.

        Args:
            configs (torch.Tensor): Joint configurations (B, C), where B is batch size, and C is 7 or 8.

        Returns:
            tuple: TorchSpheres and TorchCuboids representing the robot's configuration.
        """
        assert configs.ndim == 2
        B = configs.shape[0]
        Ms = len(SELF_COLLISION_SPHERES_HALF)
        Mc = len(self.cuboids_center)
        C = configs.shape[1]

        if C == 7:
            configs = torch.cat(
                [configs, torch.ones([B, 1], device=self.device) * self.default_prismatic_value],
                axis=1,
            )

        # right now not considering the finger dof
        link_transforms = self.compute_transformations(configs)
        fk_links = torch.cat(
            [
                link_transforms[:, 4:5].repeat(1, 2, 1, 1),
                link_transforms[:, 5:6].repeat(1, 3, 1, 1),
                link_transforms[:, 6:7].repeat(1, 5, 1, 1),
                link_transforms[:, 8:9].repeat(1, Ms + Mc - 10, 1, 1),
            ],
            dim=1,
        )

        CENTERS_ALL = CENTERS_S.copy()
        CENTERS_ALL.extend(self.cuboids_center)
        center_offsets = (
            torch.cat([torch.Tensor(CENTERS_ALL), torch.ones(Ms + Mc, 1)], dim=1)
            .to(self.device)
            .unsqueeze(0)
            .unsqueeze(-1)
        )

        centers_all = torch.matmul(fk_links, center_offsets)[:, :, :3, 0]
        sphere_centers = centers_all[:, :Ms]
        cuboid_centers = centers_all[:, Ms:]

        sphere_radii = (
            torch.Tensor(RAIUS_S).to(self.device).unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1)
        )
        cuboid_dims = torch.Tensor(self.cuboids_size).to(self.device).unsqueeze(0).repeat(B, 1, 1)

        eef_quaternions_wxyz = conversions.rotation_matrix_to_quaternion(
            link_transforms[:, 8:9, :3, :3]
        )
        cuboid_original_quaternions_wxyz = (
            torch.Tensor(self.cuboids_ori).to(self.device).unsqueeze(0)
        )
        cuboid_final_quaternions_wxyz = quaternion_multiply(
            eef_quaternions_wxyz,
            cuboid_original_quaternions_wxyz,
        )

        torch_spheres = TorchSpheres(
            sphere_centers.reshape(B, Ms, 3),
            sphere_radii.reshape(B, Ms, 1),
        )

        torch_cuboids = TorchCuboids(
            cuboid_centers.reshape(B, Mc, 3),
            cuboid_dims.reshape(B, Mc, 3),
            cuboid_final_quaternions_wxyz.reshape(B, Mc, 4),
        )

        return torch_spheres, torch_cuboids

    def check_scene_collision(
        self, config, points: np.ndarray, thred=0.001, debug=False, down_sampling: int = None
    ):
        """
        Check if there is a collision between the robot and the scene.

        Args:
            config (np.ndarray): Joint configuration.
            points (np.ndarray): Point cloud of the scene.
            thred (float): Collision checking threshold.
            debug (bool): Enable debug mode for visualization.
            down_sampling (int, optional): Number of points to downsample.

        Returns:
            bool: True if there is a collision, False otherwise.
        """
        if debug:
            t_start = time.time()
        if down_sampling is not None:
            random_indices = np.random.choice(len(points), size=down_sampling, replace=False)
            points = points[random_indices, :]
        centers, radii = self.spheres_cr(config)
        points = np.expand_dims(points, axis=2)
        ccidx = []

        centers = np.repeat(centers, points.shape[0], axis=0)
        sdf = np.linalg.norm((points - centers), axis=1) - radii

        ccidx = np.sum(sdf < thred, axis=0)
        if debug and sum(ccidx) > 0:
            t_end = time.time()
            print(f"scene collision checking time: {t_end - t_start}")
            spheres = self.spheres(config)
            scene_pcd = points[:, :, 0]
            spheres_pcd = []
            ccspheres_pcd = []
            for sphere in spheres:
                spheres_pcd.extend(sphere.sample_surface(100))
            for i in np.where(ccidx > 0)[0]:
                ccspheres_pcd.extend(spheres[i].sample_surface(100))
            spheres_pcd = np.asarray(spheres_pcd).reshape(-1, 3)
            ccspheres_pcd = np.asarray(ccspheres_pcd).reshape(-1, 3)

            spheres_rgb = np.zeros_like(spheres_pcd)
            ccspheres_rgb = np.zeros_like(ccspheres_pcd)
            scene_rgb = np.zeros_like(scene_pcd)

            spheres_rgb[:, 1] = 1
            ccspheres_rgb[:, 0] = 1
            scene_rgb[:, 2] = 1

            points = np.concatenate([spheres_pcd, ccspheres_pcd, scene_pcd])
            colors = np.concatenate([spheres_rgb, ccspheres_rgb, scene_rgb])
            save_pointcloud("neural_mp/outputs/debug_scene_cc.ply", points, colors)
            visualize_ply("neural_mp/outputs/debug_scene_cc.ply")
        return sum(ccidx) > 0

    def check_scene_collision_batch(
        self,
        configs: torch.Tensor,
        points: torch.Tensor,
        thred=0.001,
        debug=False,
        sphere_repr_only=False,
    ) -> torch.Tensor:
        """
        Check if there is a collision in the scene for a batch of configurations.

        Args:
            configs (torch.Tensor): Joint configurations.
            points (torch.Tensor): Point cloud of the scene.
            thred (float): Collision checking threshold.
            debug (bool): Enable debug mode for visualization.
            sphere_repr_only (bool): Only use sphere representation for collision checking.

        Returns:
            torch.Tensor: Number of collisions for each configuration in the batch.
        """
        sdf = self.check_scene_sdf_batch(
            configs, points, debug=debug, sphere_repr_only=sphere_repr_only
        )
        cc_num = torch.sum(sdf < thred, dim=1)
        return cc_num

    def check_scene_sdf_batch(
        self,
        configs: torch.Tensor,
        points: torch.Tensor,
        debug=False,
        sphere_repr_only=False,
    ):
        """
        check collisions of a statics scene with different robot configurations in a batch.

        Args:
            configs (torch.Tensor): Joint configurations. (B, C), B is batch size, C is 7 or 8
            points (torch.Tensor): Point cloud of the scene. (B, N, 3), B is batch size, N is number of points
            debug (bool, optional): enable pcd visualization for the spheres and cuboids (only works when batch size is 1)
            sphere_repr_only (bool, optional): only use sphere representation for collision checking

        Returns:
            torch.Tensor: Signed distance field (SDF) for the batch.
        """
        t0 = time.time()
        if sphere_repr_only:
            torch_spheres = self.torch_spheres(configs)
        else:
            torch_spheres, torch_cuboids = self.torch_spheres_cuboids(configs)

        t1 = time.time()
        if sphere_repr_only:
            sdf = torch_spheres.sdf(points)
        else:
            spheres_sdf = torch_spheres.sdf(points)
            cuboids_sdf = torch_cuboids.sdf(points)
            sdf = torch.min(cuboids_sdf, spheres_sdf)

        if debug:
            t2 = time.time()
            print(f"torch sphere generation time: {t1-t0}")
            print(f"sdf calculation time: {t2-t1}")

            scene_pcd = points[0].cpu().numpy()
            spheres_pcd = torch_spheres.sample_surface(300).reshape(-1, 3).cpu().numpy()
            cuboids_pcd = torch_cuboids.sample_surface(500).reshape(-1, 3).cpu().numpy()

            scene_rgb = np.zeros_like(scene_pcd)
            spheres_rgb = np.zeros_like(spheres_pcd)
            cuboids_rgb = np.zeros_like(cuboids_pcd)

            scene_rgb[:, 2] = 1
            spheres_rgb[:, 1] = 1
            cuboids_rgb[:, 0] = 1

            points = np.concatenate([scene_pcd, spheres_pcd, cuboids_pcd])
            colors = np.concatenate([scene_rgb, spheres_rgb, cuboids_rgb])
            save_pointcloud("neural_mp/outputs/debug_batched_scene_cc_sc.ply", points, colors)
            visualize_ply("neural_mp/outputs/debug_batched_scene_cc_sc.ply")

        return sdf

    def check_self_collision(self, config, debug=False):
        """
        Check if there is a self-collision.

        Args:
            config (np.ndarray): Joint configuration.
            debug (bool): Enable debug mode for visualization.

        Returns:
            bool: True if there is a self-collision, False otherwise.
        """
        if debug:
            t_start = time.time()
        # Cfg should have 8 dof because the two fingers mirror each other in this urdf
        cfg = np.ones(8)
        cfg[:7] = config
        cfg[-1] = self.default_prismatic_value
        fk = self.robot.link_fk(cfg, use_names=True)
        fk_points = []

        for link_name, centers in self.points:
            pc = transform_pointcloud(centers, fk[link_name], in_place=False)
            fk_points.append(pc)
        transformed_centers = np.concatenate(fk_points, axis=0)
        points_matrix = np.tile(transformed_centers, (transformed_centers.shape[0], 1, 1))
        distances = np.linalg.norm(points_matrix - points_matrix.transpose((1, 0, 2)), axis=2)

        ccpairs = np.where(distances < self.collision_matrix)
        ccidx = ccpairs[0]

        if debug and len(ccidx) > 0:
            t_end = time.time()
            print(f"self collision checking time: {t_end - t_start}")
            spheres = self.spheres(config)
            spheres_pcd = []
            ccspheres_pcd = []
            for sphere in spheres:
                spheres_pcd.extend(sphere.sample_surface(100))
            for i in ccidx:
                ccspheres_pcd.extend(spheres[i].sample_surface(100))
            spheres_pcd = np.asarray(spheres_pcd).reshape(-1, 3)
            ccspheres_pcd = np.asarray(ccspheres_pcd).reshape(-1, 3)

            spheres_rgb = np.zeros_like(spheres_pcd)
            ccspheres_rgb = np.zeros_like(ccspheres_pcd)

            spheres_rgb[:, 1] = 1
            ccspheres_rgb[:, 0] = 1

            points = np.concatenate([spheres_pcd, ccspheres_pcd])
            colors = np.concatenate([spheres_rgb, ccspheres_rgb])
            save_pointcloud("neural_mp/outputs/debug_scene_cc.ply", points, colors)
            visualize_ply("neural_mp/outputs/debug_scene_cc.ply")
        return len(ccidx) > 0

    def check_collision(
        self, config, points: np.ndarray, thred=0.001, debug=False, down_sampling: int = None
    ):
        """
        Check if there is a collision in the scene or self-collision.

        Args:
            config (np.ndarray): Joint configuration.
            points (np.ndarray): Point cloud of the scene.
            thred (float): Collision checking threshold.
            debug (bool): Enable debug mode for visualization.
            down_sampling (int, optional): Number of points to downsample.

        Returns:
            bool: True if there is a collision, False otherwise.
        """
        return self.check_scene_collision(
            config, points, thred=thred, debug=debug, down_sampling=down_sampling
        ) or self.check_self_collision(config, debug=debug)

    def check_collision_traj(
        self, configs, points: np.ndarray, thred=0.001, debug=False, down_sampling: int = None
    ):
        """
        Check the number of collisions in a given trajectory.

        Args:
            configs (list): List of joint configurations in the trajectory.
            points (np.ndarray): Point cloud of the scene.
            thred (float): Collision checking threshold.
            debug (bool): Enable debug mode for visualization.
            down_sampling (int, optional): Number of points to downsample.

        Returns:
            int: Number of collisions in the trajectory.
        """
        num_collisions = 0
        for config in configs:
            if self.check_collision(
                config, points, thred=thred, debug=debug, down_sampling=down_sampling
            ):
                num_collisions += 1
        return num_collisions
