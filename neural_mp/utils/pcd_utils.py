import cv2
import numpy as np
import pybullet as p
import torch
from geometrout.primitive import Cuboid, Cylinder, Sphere

from neural_mp.utils.geometry import ObjaMesh, construct_mixed_point_cloud
from robofin.pointcloud.torch import FrankaSampler
from robofin.robots import FrankaRobot

fk_sampler = FrankaSampler(
    "cpu",
    use_cache=True,
    num_fixed_points=4096,
    default_prismatic_value=0.025,
)


def compute_scene_pcd_params(
    max_num_objs_per_type,
    cuboid_dims,
    cuboid_centers,
    cuboid_quats,
    cylinder_radii,
    cylinder_heights,
    cylinder_centers,
    cylinder_quats,
    sphere_centers,
    sphere_radii,
    mesh_position,
    mesh_scale,
    mesh_quaternion,
    obj_id,
    mesh_id,
):
    """
    Compute the scene pcd params, making sure to pad with zeros.
    Args:
        max_num_objs_per_type (int): maximum number of objects per type.
        cuboid_dims (np.array): array of cuboid dimensions.
        cuboid_centers (np.array): array of cuboid centers.
        cuboid_quats (np.array): array of cuboid quaternions.
        cylinder_radii (np.array): array of cylinder radii.
        cylinder_heights (np.array): array of cylinder heights.
        cylinder_centers (np.array): array of cylinder centers.
        cylinder_quats (np.array): array of cylinder quaternions.
        sphere_centers (np.array): array of sphere centers.
        sphere_radii (np.array): array of sphere radii.
    """
    M = max_num_objs_per_type
    num_cuboids = cuboid_dims.shape[0] // 3
    num_cylinders = cylinder_radii.shape[0]
    num_spheres = sphere_radii.shape[0]
    num_meshes = mesh_id.shape[0]
    cuboid_params = np.concatenate(
        [
            cuboid_dims,
            np.zeros((M - num_cuboids) * 3, dtype=np.float32),
            cuboid_centers,
            np.zeros((M - num_cuboids) * 3, dtype=np.float32),
            cuboid_quats,
            np.zeros((M - num_cuboids) * 4, dtype=np.float32),
        ]
    )
    cylinder_params = np.concatenate(
        [
            cylinder_radii,
            np.zeros((M - num_cylinders) * 1, dtype=np.float32),
            cylinder_heights,
            np.zeros((M - num_cylinders) * 1, dtype=np.float32),
            cylinder_centers,
            np.zeros((M - num_cylinders) * 3, dtype=np.float32),
            cylinder_quats,
            np.zeros((M - num_cylinders) * 4, dtype=np.float32),
        ]
    )
    sphere_params = np.concatenate(
        [
            sphere_centers,
            np.zeros((M - num_spheres) * 3, dtype=np.float32),
            sphere_radii,
            np.zeros((M - num_spheres) * 1, dtype=np.float32),
        ]
    )

    mesh_params = np.concatenate(
        [
            mesh_position,
            np.zeros((M - num_meshes) * 3, dtype=np.float32),
            mesh_scale,
            np.zeros((M - num_meshes) * 1, dtype=np.float32),
            mesh_quaternion,
            np.zeros((M - num_meshes) * 4, dtype=np.float32),
            obj_id,
            np.zeros((M - num_meshes) * 1, dtype=np.float32),
            mesh_id,
            np.zeros((M - num_meshes) * 1, dtype=np.float32),
        ]
    )

    scene_pcd_params = np.concatenate(
        [
            np.array([M]).astype(np.float32),
            cuboid_params,
            cylinder_params,
            sphere_params,
            mesh_params,
        ]
    )
    assert scene_pcd_params.dtype == np.float32, scene_pcd_params.dtype
    return scene_pcd_params


def compute_scene_pcd_params_batched(
    max_num_objs_per_type,
    cuboid_dims,
    cuboid_centers,
    cuboid_quats,
    cylinder_radii,
    cylinder_heights,
    cylinder_centers,
    cylinder_quats,
    sphere_centers,
    sphere_radii,
):
    """
    Compute the scene pcd params, making sure to pad with zeros.
    Args:
        max_num_objs_per_type (int): maximum number of objects per type.
        cuboid_dims (torch.Tensor): array of cuboid dimensions.
        cuboid_centers (torch.Tensor): array of cuboid centers.
        cuboid_quats (torch.Tensor): array of cuboid quaternions.
        cylinder_radii (torch.Tensor): array of cylinder radii.
        cylinder_heights (torch.Tensor): array of cylinder heights.
        cylinder_centers (torch.Tensor): array of cylinder centers.
        cylinder_quats (torch.Tensor): array of cylinder quaternions.
        sphere_centers (torch.Tensor): array of sphere centers.
        sphere_radii (torch.Tensor): array of sphere radii.
    """
    M = max_num_objs_per_type
    B = cuboid_dims.shape[0]
    device = cuboid_dims.device
    num_cuboids = cuboid_dims.shape[1] // 3
    num_cylinders = cylinder_radii.shape[1]
    num_spheres = sphere_radii.shape[1]
    cuboid_params = torch.cat(
        (
            cuboid_dims,
            torch.zeros(B, (M - num_cuboids) * 3, device=device),
            cuboid_centers,
            torch.zeros(B, (M - num_cuboids) * 3, device=device),
            cuboid_quats,
            torch.zeros(B, (M - num_cuboids) * 4, device=device),
        ),
        dim=1,
    )
    cylinder_params = torch.cat(
        (
            cylinder_radii,
            torch.zeros(B, (M - num_cylinders) * 1, device=device),
            cylinder_heights,
            torch.zeros(B, (M - num_cylinders) * 1, device=device),
            cylinder_centers,
            torch.zeros(B, (M - num_cylinders) * 3, device=device),
            cylinder_quats,
            torch.zeros(B, (M - num_cylinders) * 4, device=device),
        ),
        dim=1,
    )
    sphere_params = torch.cat(
        (
            sphere_centers,
            torch.zeros(B, (M - num_spheres) * 3, device=device),
            sphere_radii,
            torch.zeros(B, (M - num_spheres) * 1, device=device),
        ),
        dim=1,
    )
    scene_pcd_params = torch.cat(
        (
            torch.Tensor([[M], [M]]),
            cuboid_params,
            cylinder_params,
            sphere_params,
        ),
        dim=1,
    )
    return scene_pcd_params


def create_obstacle_list(
    cuboid_dims,
    cuboid_centers,
    cuboid_quats,
    cylinder_radii,
    cylinder_heights,
    cylinder_centers,
    cylinder_quats,
    sphere_centers,
    sphere_radii,
    table_dims=[2, 2, 0.01],
    table_center=[0, 0, 0.005],
    table_quat=[1, 0, 0, 0],
):
    """
    Create a list of obstacles from the given parameters. Note: input quaternion in wxyz format
    """
    num_cuboids = cuboid_dims.shape[0] // 3
    num_cylinders = cylinder_radii.shape[0]
    num_spheres = sphere_radii.shape[0]

    obstacles = []
    # add table
    obstacles.append(Cuboid(table_center, table_dims, table_quat))

    if num_cuboids > 0:
        cuboid_dims = cuboid_dims.reshape(-1, 3)
        cuboid_centers = cuboid_centers.reshape(-1, 3)
        cuboid_centers[:, 2] += cuboid_dims[:, 2] / 2
        cuboid_quats = cuboid_quats.reshape(-1, 4)
        for i in range(cuboid_dims.shape[0]):
            obstacles.append(Cuboid(cuboid_centers[i], cuboid_dims[i], cuboid_quats[i]))

    if num_cylinders > 0:
        cylinder_radii = cylinder_radii.reshape(-1)
        cylinder_heights = cylinder_heights.reshape(-1)
        cylinder_centers = cylinder_centers.reshape(-1, 3)
        cylinder_centers[:, 2] += cylinder_heights / 2
        cylinder_quats = cylinder_quats.reshape(-1, 4)
        for i in range(cylinder_radii.shape[0]):
            obstacles.append(
                Cylinder(
                    cylinder_centers[i], cylinder_radii[i], cylinder_heights[i], cylinder_quats[i]
                )
            )

    if num_spheres > 0:
        sphere_radii = sphere_radii.reshape(-1)
        sphere_centers = sphere_centers.reshape(-1, 3)
        sphere_centers[:, 2] += sphere_radii
        for i in range(sphere_radii.shape[0]):
            obstacles.append(Sphere(sphere_centers[i], sphere_radii[i]))
    return obstacles


def decompose_pcd_params_obs(pcd_params_obs):
    """
    Decompose the pcd params observation.
    """
    current_joint_angles, goal_joint_angles, scene_pcd_params = np.split(pcd_params_obs, [7, 14])
    goal_joint_angles, gripper_state = np.split(goal_joint_angles, [7])
    return (
        current_joint_angles,
        goal_joint_angles,
        gripper_state,
        *decompose_scene_pcd_params_obs(scene_pcd_params),
    )


def decompose_scene_pcd_params_obs(scene_pcd_params):
    """
    Decompose the pcd params observation.
    """
    M = int(scene_pcd_params[0])

    cuboid_params = scene_pcd_params[1 : 1 + 10 * M]
    cuboid_dims = cuboid_params[: 3 * M].reshape(-1, 3)
    cuboid_centers = cuboid_params[3 * M : 6 * M].reshape(-1, 3)
    cuboid_quats = cuboid_params[6 * M : 10 * M].reshape(-1, 4)

    cylinder_params = scene_pcd_params[1 + 10 * M : 1 + 10 * M + 9 * M]
    cylinder_radii = cylinder_params[: 1 * M].reshape(-1)
    cylinder_heights = cylinder_params[1 * M : 2 * M].reshape(-1)
    cylinder_centers = cylinder_params[2 * M : 5 * M].reshape(-1, 3)
    cylinder_quats = cylinder_params[5 * M : 9 * M].reshape(-1, 4)

    sphere_params = scene_pcd_params[1 + 10 * M + 9 * M : 1 + 10 * M + 9 * M + 4 * M]
    sphere_centers = sphere_params[: 3 * M].reshape(-1, 3)
    sphere_radii = sphere_params[3 * M : 4 * M].reshape(-1)

    mesh_params = scene_pcd_params[1 + 10 * M + 9 * M + 4 * M :]
    mesh_positions = mesh_params[: 3 * M].reshape(-1, 3)
    mesh_scales = mesh_params[3 * M : 4 * M].reshape(-1)
    mesh_quaternions = mesh_params[4 * M : 8 * M].reshape(-1, 4)
    obj_ids = mesh_params[8 * M : 9 * M].reshape(-1)
    mesh_ids = mesh_params[9 * M : 10 * M].reshape(-1)

    return (
        np.array(cuboid_dims).astype(np.float32),
        np.array(cuboid_centers).astype(np.float32),
        np.array(cuboid_quats).astype(np.float32),
        np.array(cylinder_radii).astype(np.float32),
        np.array(cylinder_heights).astype(np.float32),
        np.array(cylinder_centers).astype(np.float32),
        np.array(cylinder_quats).astype(np.float32),
        np.array(sphere_centers).astype(np.float32),
        np.array(sphere_radii).astype(np.float32),
        np.array(
            mesh_positions,
        ).astype(np.float32),
        np.array(
            mesh_scales,
        ).astype(np.float32),
        np.array(
            mesh_quaternions,
        ).astype(np.float32),
        np.array(
            obj_ids,
        ).astype(np.float32),
        np.array(
            mesh_ids,
        ).astype(np.float32),
    )


def decompose_scene_pcd_params_obs_batched(scene_pcd_params):
    """
    Decompose the pcd params observation.
    """
    M = scene_pcd_params[:, 0].int().max().item()
    cuboid_params = scene_pcd_params[:, 1 : 1 + 10 * M]

    cuboid_dims = cuboid_params[:, : 3 * M]
    cuboid_centers = cuboid_params[:, 3 * M : 6 * M]
    cuboid_quats = cuboid_params[:, 6 * M : 10 * M]

    cylinder_params = scene_pcd_params[:, 1 + 10 * M : 1 + 10 * M + 9 * M]
    cylinder_radii = cylinder_params[:, : 1 * M]
    cylinder_heights = cylinder_params[:, 1 * M : 2 * M]
    cylinder_centers = cylinder_params[:, 2 * M : 5 * M]
    cylinder_quats = cylinder_params[:, 5 * M : 9 * M]

    sphere_params = scene_pcd_params[:, 1 + 10 * M + 9 * M :]
    sphere_centers = sphere_params[:, : 3 * M]
    sphere_radii = sphere_params[:, 3 * M : 4 * M]

    mesh_start_index = 1 + 10 * M + 9 * M + 4 * M
    mesh_params = scene_pcd_params[:, mesh_start_index : mesh_start_index + 12 * M]
    mesh_positions = mesh_params[:, : 3 * M]
    mesh_scales = mesh_params[:, 3 * M : 6 * M]
    mesh_quaternions = mesh_params[:, 6 * M : 10 * M]
    obj_ids = mesh_params[:, 10 * M : 11 * M]
    mesh_ids = mesh_params[:, 11 * M : 12 * M]

    return (
        cuboid_dims,
        cuboid_centers,
        cuboid_quats,
        cylinder_radii,
        cylinder_heights,
        cylinder_centers,
        cylinder_quats,
        sphere_centers,
        sphere_radii,
        mesh_positions,
        mesh_scales,
        mesh_quaternions,
        obj_ids,
        mesh_ids,
        M,
    )


def compute_scene_oracle_pcd(
    num_obstacle_points,
    cuboid_dims,
    cuboid_centers,
    cuboid_quats,
    cylinder_radii,
    cylinder_heights,
    cylinder_centers,
    cylinder_quats,
    sphere_centers,
    sphere_radii,
    mesh_position,
    mesh_scale,
    mesh_quaternion,
    obj_id,
    mesh_id,
):
    """
    Compute the oracle point cloud.
    """

    def quaternions_xyzw_to_wxyz(quaternions_xyzw):
        quaternions_wxyz = quaternions_xyzw[:, [3, 0, 1, 2]]
        return quaternions_wxyz

    cuboids = [
        Cuboid(c, d, q)
        for c, d, q in zip(cuboid_centers, cuboid_dims, quaternions_xyzw_to_wxyz(cuboid_quats))
    ]
    cuboids = [c for c in cuboids if not c.is_zero_volume()]
    cylinders = [
        Cylinder(c, r, h, q)
        for c, r, h, q in zip(
            cylinder_centers,
            cylinder_radii,
            cylinder_heights,
            quaternions_xyzw_to_wxyz(cylinder_quats),
        )
    ]
    cylinders = [c for c in cylinders if not c.is_zero_volume()]

    spheres = [Sphere(c, r) for c, r in zip(sphere_centers, sphere_radii)]
    spheres = [s for s in spheres if not s.is_zero_volume()]
    meshes = [
        ObjaMesh(pos, scale, quat, obj_id, str(int(mesh_id)))
        for pos, scale, quat, obj_id, mesh_id in zip(
            mesh_position, mesh_scale, quaternions_xyzw_to_wxyz(mesh_quaternion), obj_id, mesh_id
        )
        if obj_id != 0.0 and scale > 0
    ]
    meshes = [m for m in meshes if not m.is_zero_volume()]

    obstacle_points = construct_mixed_point_cloud(
        cuboids + cylinders + spheres + meshes, num_obstacle_points
    )[:, :3]
    return obstacle_points


def compute_in_hand_pcd(joint_angles, num_points, in_hand_params):
    """
    Compute the in hand pcd.
    """
    in_hand_pcds = []
    for joint_angle in joint_angles:
        eef_pose_SE3 = FrankaRobot.fk(joint_angle, eff_frame="panda_link8")
        eef_pos = np.asarray(eef_pose_SE3.xyz)
        eef_ori = np.asarray(eef_pose_SE3.so3.xyzw)

        in_hand_type = ["box", "cylinder", "sphere", "mesh"][int(in_hand_params[0])]
        in_hand_size = in_hand_params[1:4]
        in_hand_pos = in_hand_params[4:7]
        in_hand_ori = in_hand_params[7:11]

        global_pos, global_ori = p.multiplyTransforms(eef_pos, eef_ori, in_hand_pos, in_hand_ori)

        global_pos = np.asarray(global_pos)
        global_ori = np.asarray(global_ori)

        if in_hand_type == "box":
            in_hand_obj = [Cuboid(global_pos, in_hand_size, global_ori)]
        elif in_hand_type == "cylinder":
            in_hand_obj = [Cylinder(global_pos, in_hand_size[0], in_hand_size[1], global_ori)]
        elif in_hand_type == "sphere":
            in_hand_obj = [Sphere(global_pos, in_hand_size[0])]
        elif in_hand_type == "mesh":
            in_hand_obj = [
                ObjaMesh(
                    global_pos,
                    in_hand_size[0],
                    global_ori,
                    in_hand_size[1],
                    str(int(in_hand_size[2])),
                )
            ]

        in_hand_pcd = construct_mixed_point_cloud(in_hand_obj, num_points)[:, :3]
        in_hand_pcds.append(in_hand_pcd)
    return np.asarray(in_hand_pcds)


def vectorized_subsample(inputs, dim=1, num_points=2048):
    batch_size = inputs.shape[0]
    random_indices = torch.randint(
        0, inputs.shape[dim], (batch_size, num_points), device=inputs.device
    )
    batch_indices = (
        torch.arange(batch_size, device=inputs.device).unsqueeze(1).expand(-1, num_points)
    )
    inputs = inputs[batch_indices, random_indices, :]
    return inputs


def has_object_in_hand(scene_pcd_params):
    """Is there object in hand?"""
    if not isinstance(scene_pcd_params, np.ndarray) or scene_pcd_params.size == 0:
        return False

    total_params = len(scene_pcd_params)
    points_per_object = scene_pcd_params[0]
    expected_params_with_object = 33 * points_per_object + 11

    return total_params == expected_params_with_object


def compute_full_pcd(
    pcd_params,
    num_robot_points,
    num_obstacle_points,
    num_in_hand_points=500,
    target_pcd_type="joint",
):
    """
    Compute the full pcd of the scene + robot with optimizations.
    Args:
        pcd_params (np.ndarray): (B, T, ) array of pcd params.
        num_robot_points (int): Number of points to sample from the robot.
        num_obstacle_points (int): Number of points to sample from the obstacles.
    """
    joint_angles, goal_angles, gripper_state, scene_pcd_params = (
        pcd_params[:, :7],
        pcd_params[:, 7:14],
        pcd_params[:, 14:15],
        pcd_params[:, 15:],
    )

    scene_pcd_params = scene_pcd_params[0]  # Assuming scene does not change

    robot_pcd = fk_sampler.sample(
        torch.from_numpy(np.concatenate((joint_angles, gripper_state), axis=1))
    )

    has_in_hand = has_object_in_hand(scene_pcd_params)

    if has_in_hand:
        in_hand_params = scene_pcd_params[-11:]
        in_hand_pcd = compute_in_hand_pcd(joint_angles, num_in_hand_points, in_hand_params)
        scene_pcd_params = scene_pcd_params[:-11]
        robot_pcd = torch.cat((robot_pcd, torch.from_numpy(in_hand_pcd)), dim=1)
    decomposed_scene_pcd_params = decompose_scene_pcd_params_obs(scene_pcd_params)
    scene_pcd = compute_scene_oracle_pcd(num_obstacle_points, *decomposed_scene_pcd_params)
    scene_pcd = np.tile(scene_pcd, (pcd_params.shape[0], 1, 1))

    robot_pcd = vectorized_subsample(robot_pcd, dim=1, num_points=num_robot_points).numpy()

    if target_pcd_type == "joint":
        target_pcd = fk_sampler.sample(
            torch.from_numpy(np.concatenate((goal_angles, gripper_state), axis=1))
        )
        if has_in_hand:
            target_in_hand_pcd = compute_in_hand_pcd(
                goal_angles, num_in_hand_points, in_hand_params
            )
            target_pcd = torch.cat((target_pcd, torch.from_numpy(target_in_hand_pcd)), dim=1)
        target_pcd = vectorized_subsample(target_pcd, dim=1, num_points=num_robot_points).numpy()
    elif target_pcd_type == "ee":
        target_robot_ee = fk_sampler.end_effector_pose(torch.from_numpy(goal_angles))
        target_pcd = fk_sampler.sample_end_effector(target_robot_ee, num_points=256)
    else:
        raise ValueError("Invalid target_pcd_type")
    robot_pcd = np.concatenate(
        [robot_pcd, np.zeros((robot_pcd.shape[0], robot_pcd.shape[1], 1))], axis=-1
    )
    scene_pcd = np.concatenate(
        [scene_pcd, np.ones((scene_pcd.shape[0], scene_pcd.shape[1], 1))], axis=-1
    )
    target_pcd = np.concatenate(
        [target_pcd, 2 * np.ones((target_pcd.shape[0], target_pcd.shape[1], 1))],
        axis=-1,
    )
    return np.concatenate([robot_pcd, scene_pcd, target_pcd], axis=1).astype(np.float32)


def discretize_depth(cv_depth):
    min_val = np.min(cv_depth)
    max_val = np.max(cv_depth)
    depth_range = max_val - min_val
    depth8 = (255.0 / depth_range * (cv_depth - min_val)).astype("uint8")
    return depth8


def depth_to_rgb(depth):
    """
    depth in [-1, 1]
    out in [0, 255]
    """
    depth = depth.astype(np.uint8)
    depth8_rgb = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
    depth_color = cv2.applyColorMap(depth8_rgb, cv2.COLORMAP_JET)
    return depth_color
