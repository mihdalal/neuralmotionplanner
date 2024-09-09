# MIT License
#
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, University of Washington. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import json
import os
import random
from typing import Sequence, Union

import numpy as np
import torch
from geometrout.primitive import Cuboid, Cylinder, Sphere
from geometrout.transform import SE3, SO3
from pytorch3d.renderer import (
    AlphaCompositor,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    look_at_view_transform,
)
from pytorch3d.structures import Pointclouds
from scipy.spatial import ConvexHull

from robofin.pointcloud.torch import FrankaSampler


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


def render_single_pointcloud(pcd, image_size=512):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    verts = torch.Tensor(pcd[:, :3]).to(device)

    rgb_pcd = torch.ones_like(verts) * torch.tensor([255, 0, 0]).to(device)
    rgb = torch.Tensor(rgb_pcd).to(device)

    point_cloud = Pointclouds(points=[verts], features=[rgb])

    # Initialize a camera.
    R, T = look_at_view_transform(1, 0, 90, up=((1, 0, 0),))
    cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters.
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        # radius = 0.003,
        points_per_pixel=10,
    )

    # Create a points renderer by compositing points using an alpha compositor (nearer points
    # are weighted more heavily). See [1] for an explanation.
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())

    images = renderer(point_cloud)
    img = images[0, ..., :3].cpu().numpy()
    return img


def transform_points(pc, transformation_matrix, in_place=True):
    """
    Transform points using a transformation matrix in PyTorch.

    Parameters
    ----------
    :param pc: A PyTorch tensor representing a pointcloud.
        This should have shape B x N x [3 + M] where N is the number of points. B is the batch size
    :param transformation_matrix: A Bx4x4 homography tensor.
    :param in_place: A flag indicating whether to perform the operation in-place.

    :return: A transformed pointcloud tensor.
    """
    B, M, N = pc.shape[:-1]
    xyz = pc[..., :3]
    ones = torch.ones(*xyz.shape[:-1], 1, dtype=xyz.dtype, device=xyz.device)
    homogeneous_xyz = torch.cat((xyz, ones), dim=-1)
    transformed_xyz = torch.matmul(transformation_matrix, homogeneous_xyz.transpose(3, 2))

    if in_place:
        pc[:] = transformed_xyz[..., :3, :].transpose(3, 2)
        return pc
    else:
        return torch.cat((transformed_xyz[..., :3, :].transpose(3, 2), pc[..., 3:]), dim=1)


def quats_to_rot(quats: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of quaternions to a batch of rotation matrices.

    Args:
        quats (torch.Tensor): A tensor of quaternions, has dim [B, M, 4]

    Returns:
        torch.Tensor: A tensor of rotation matrices, has dim [B, M, 3, 3]
    """
    # Ensure quaternion is normalized
    quats = quats / quats.norm(dim=-1, keepdim=True)

    # Extract the values
    w, x, y, z = quats[..., 0], quats[..., 1], quats[..., 2], quats[..., 3]

    # Compute rotation matrix components
    xx, yy, zz = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    # Assemble the rotation matrix
    rotMat = torch.empty(quats.size()[:-1] + (3, 3), device=quats.device, dtype=quats.dtype)
    rotMat[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    rotMat[..., 0, 1] = 2.0 * (xy - wz)
    rotMat[..., 0, 2] = 2.0 * (xz + wy)
    rotMat[..., 1, 0] = 2.0 * (xy + wz)
    rotMat[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    rotMat[..., 1, 2] = 2.0 * (yz - wx)
    rotMat[..., 2, 0] = 2.0 * (xz - wy)
    rotMat[..., 2, 1] = 2.0 * (yz + wx)
    rotMat[..., 2, 2] = 1.0 - 2.0 * (xx + yy)

    return rotMat


class TorchSpheres:
    """
    A Pytorch representation of a batch of M spheres (i.e. B elements in the batch,
    M spheres per element). Any of these spheres can have zero volume (these
    will be masked out during calculation of the various functions in this
    class, such as sdf).
    """

    def __init__(self, centers: torch.Tensor, radii: torch.Tensor):
        """
        :param centers torch.Tensor: a set of centers, has dim [B, M, 3]
        :param radii torch.Tensor: a set of radii, has dim [B, M, 1]
        """
        assert centers.ndim == 3
        assert radii.ndim == 3

        # TODO It would be more memory efficient to rely more heavily on broadcasting
        # in some cases where multiple spheres have the same size
        assert centers.ndim == radii.ndim

        # This logic is to determine the batch size. Either batch sizes need to
        # match or if only one variable has a batch size, that one is assumed to
        # be the batch size

        self.centers = centers
        self.radii = radii
        self.mask = ~torch.isclose(self.radii, torch.zeros(1).type_as(centers)).squeeze(-1)

    def surface_area(self) -> torch.Tensor:
        """
        Calculates the surface area of the spheres

        :rtype torch.Tensor: A tensor of the surface areas of the spheres
        """
        area = 4 * np.pi * torch.pow(self.radii, 3)
        return area.squeeze(-1)

    def sample_surface(self, num_points: int, noise: float = 0.0) -> torch.Tensor:
        """
        Samples points from all spheres, including ones with zero volume

        :param num_points int: The number of points to sample per sphere
        :rtype torch.Tensor: The points, has dim [B, M, N]
        """
        B, M, _ = self.centers.shape
        unnormalized_points = torch.rand((B, M, num_points, 3), device=self.centers.device) * 2 - 1
        normalized = (
            unnormalized_points / torch.linalg.norm(unnormalized_points, dim=-1)[:, :, :, None]
        )
        random_points = normalized * self.radii[:, :, None, :] + self.centers[:, :, None, :]
        random_points += (torch.rand_like(random_points) * 2 - 1) * noise
        return random_points

    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        """
        :param points torch.Tensor: The points with which to calculate the
                                    SDF, has dim [B, N, 3] (N is the number of points)
        :rtype torch.Tensor: The scene SDF value for each point (i.e. the minimum SDF
                             value for each of the M spheres), has dim [B, N]
        """
        # merging everything into one line to make it memory efficient
        return torch.min(
            torch.linalg.norm(points[:, None, :, :] - self.centers[:, :, None, :], dim=-1)
            - self.radii,
            dim=1,
        )[0]

    def sdf_sequence(self, points: torch.Tensor) -> torch.Tensor:
        """
        Calculates SDF values for a time sequence of point clouds
        :param points torch.Tensor: The batched sequence of point clouds with
                                    dimension [B, T, N, 3] (T in sequence length,
                                    N is number of points)
        :rtype torch.Tensor: The scene SDF for each point at each timestep (i.e. the minimum
                             SDF value across the M spheres at each timestep),
                             has dim [B, T, N]
        """
        assert points.ndim == 4
        B, M, _ = self.radii.shape
        _, T, N, _ = points.shape
        all_sdfs = float("inf") * torch.ones(B, M, T, N).type_as(points)
        distances = points[:, None, :, :] - self.centers[:, :, None, None, :]
        all_sdfs[self.mask] = (
            torch.linalg.norm(distances[self.mask], dim=-1) - self.radii[:, :, None, :][self.mask]
        )
        return torch.min(all_sdfs, dim=1)[0]


class TorchCuboids:
    """
    A Pytorch representation of a batch of M cuboids (i.e. B elements in the batch,
    M cuboids per element). Any of these cuboids can have zero volume (these
    will be masked out during calculation of the various functions in this
    class, such as sdf).
    """

    def __init__(self, centers: torch.Tensor, dims: torch.Tensor, quaternions: torch.Tensor):
        """
        :param centers torch.Tensor: Has dim [B, M, 3]
        :param dims torch.Tensor: Has dim [B, M, 3]
        :param quaternions torch.Tensor: Has dim [B, M, 4] with quaternions formatted as
                                   w, x, y, z
        """
        assert centers.ndim == 3
        assert dims.ndim == 3
        assert quaternions.ndim == 3

        self.dims = dims
        self.centers = centers
        # It's helpful to ensure the quaternions are normalized
        self.quats = quaternions / torch.linalg.norm(quaternions, dim=2)[:, :, None]

        self.pose_4x4 = torch.zeros((centers.size(0), centers.size(1), 4, 4)).type_as(centers)
        self.pose_4x4[:, :, 3, 3] = 1
        self.pose_4x4[:, :, :3, :3] = quats_to_rot(self.quats)
        self.pose_4x4[:, :, :3, 3] = self.centers

        self._init_frames()
        # Mask for nonzero volumes
        self.mask = ~torch.any(torch.isclose(self.dims, torch.zeros(1).type_as(centers)), dim=-1)

    def surface_area(self) -> torch.Tensor:
        """
        Calculates the surface area of the cuboids

        :rtype torch.Tensor: A tensor of the surface areas of the cuboids
        """
        area = 2 * (
            self.dims[:, :, 0] * self.dims[:, :, 1]
            + self.dims[:, :, 0] * self.dims[:, :, 2]
            + self.dims[:, :, 1] * self.dims[:, :, 2]
        )
        return area

    def sample_surface(self, num_points, noise=0.0):
        """
        Samples random points on the surface of the cube. Probabilities are
        weighed based on area of each side.

        :param num_points: The number of points to sample on the surface
        :param noise: The range of uniform noise to apply to samples

        :return: A random pointcloud sampled from the surface of the cuboid
        """
        B, M, _ = self.dims.shape  # Batch size, Number of cuboids, Dimensions

        # Sample random points within a unit cube
        random_points = (
            torch.rand((B, M, num_points, 3), device=self.dims.device, dtype=self.dims.dtype) * 2
            - 1
        )

        # Scale the points according to the cuboid dimensions
        scaled_points = random_points * self.dims.unsqueeze(2) / 2

        # Calculate the surface area of each face
        face_areas = torch.stack(
            [
                self.dims[:, :, 1] * self.dims[:, :, 2],
                self.dims[:, :, 0] * self.dims[:, :, 2],
                self.dims[:, :, 0] * self.dims[:, :, 1],
            ],
            dim=-1,
        ).repeat(
            1, 1, 2
        )  # Each face area appears twice

        # Compute probabilities for each face
        face_probs = face_areas / face_areas.sum(dim=-1, keepdim=True)

        # throwout nans
        face_probs[torch.isnan(face_probs)] = 0

        # replace any row in the last dim of probs that is all 0s with 1/6
        face_probs[torch.all(face_probs == 0, dim=-1)] = 1 / 6

        # Choose faces based on probabilities
        chosen_faces = torch.multinomial(face_probs.view(B * M, -1), num_points, replacement=True)

        # Project points onto the chosen faces
        new_scaled_points = torch.zeros_like(scaled_points)
        for i in range(6):
            mask = ((chosen_faces == i).view(B, M, num_points)).unsqueeze(-1)
            dim_array = torch.eye(3, device=self.dims.device)[i % 3].repeat(B, M, num_points, 1)
            new_scaled_points += scaled_points * (mask * (1 - dim_array)) + (
                self.dims[:, :, i % 3] / 2 * (-1 if i // 3 else 1)
            ).unsqueeze(-1).unsqueeze(-1) * (mask * dim_array)
        scaled_points = new_scaled_points
        # Transform points to world coordinates
        transformed_points = transform_points(scaled_points, self.pose_4x4)

        # Add noise
        transformed_points += (torch.rand_like(transformed_points) * 2 - 1) * noise

        # throwout nans
        transformed_points[torch.isnan(transformed_points)] = 0

        return transformed_points.view(B, M, num_points, 3)

    def geometrout(self):
        """
        Helper method to convert this into geometrout primitives
        """
        B, M, _ = self.centers.shape
        return [
            [
                Cuboid(
                    center=self.centers[bidx, midx, :].detach().cpu().numpy(),
                    dims=self.dims[bidx, midx, :].detach().cpu().numpy(),
                    quaternion=self.quats[bidx, midx, :].detach().cpu().numpy(),
                )
                for midx in range(M)
                if self.mask[bidx, midx]
            ]
            for bidx in range(B)
        ]

    def _init_frames(self):
        """
        In order to calculate the SDF, we need to calculate the inverse
        transformation of the cuboid. This is because we are transforming points
        in the world frame into the cuboid frame.
        """
        B = self.centers.size(0)
        M = self.centers.size(1)

        R = quats_to_rot(-self.quats)
        t = self.centers

        R_inv = R.transpose(-2, -1)

        t_inv = -torch.matmul(R_inv, t.unsqueeze(-1)).squeeze(-1)

        self.inv_frames = torch.zeros((B, M, 4, 4)).type_as(self.centers)
        self.inv_frames[:, :, :3, :3] = R_inv
        self.inv_frames[:, :, :3, 3] = t_inv
        self.inv_frames[:, :, 3, 3] = 1
        self.inv_frames[torch.isnan(self.inv_frames)] = 0

    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        """
        :param points torch.Tensor: The points with which to calculate the SDF, has
                                    dim [B, N, 3] (N is the number of points)
        :rtype torch.Tensor: The scene SDF value for each point (i.e. the minimum SDF
                             value for each of the M cuboids), has dim [B, N]
        """
        assert points.ndim == 3
        # We are going to need to map points in the global frame to the cuboid frame
        # First take the points and make them homogeneous by adding a one to the end

        # points_from_volumes = points[self.nonzero_volumes, :, :]
        assert points.size(0) == self.centers.size(0)
        if torch.all(~self.mask):
            return float("inf") * torch.ones(points.size(0), points.size(1)).type_as(points)

        homog_points = torch.cat(
            (
                points,
                torch.ones((points.size(0), points.size(1), 1)).type_as(points),
            ),
            dim=2,
        )
        # Next, project these points into their respective cuboid frames
        # Will return [B x M x N x 3]

        points_proj = torch.matmul(
            self.inv_frames[:, :, None, :, :], homog_points[:, None, :, :, None]
        ).squeeze(-1)[:, :, :, :3]
        B, M, N, _ = points_proj.shape
        masked_points = points_proj[self.mask]

        # The follow computations are adapted from here
        # https://github.com/fogleman/sdf/blob/main/sdf/d3.py
        # Move points to top corner

        distances = torch.abs(masked_points) - (self.dims[self.mask] / 2)[:, None, :]
        # This is distance only for points outside the box, all points inside return zero
        # This probably needs to be fixed or else there will be a nan gradient
        outside = torch.linalg.norm(torch.maximum(distances, torch.zeros_like(distances)), dim=-1)
        # This is distance for points inside the box, all others return zero
        inner_max_distance = torch.max(distances, dim=-1).values
        inside = torch.minimum(inner_max_distance, torch.zeros_like(inner_max_distance))
        all_sdfs = float("inf") * torch.ones(B, M, N).type_as(points)
        all_sdfs[self.mask] = outside + inside
        return torch.min(all_sdfs, dim=1)[0]

    def sdf_sequence(self, points: torch.Tensor) -> torch.Tensor:
        """
        Calculates SDF values for a time sequence of point clouds
        :param points torch.Tensor: The batched sequence of point clouds with
                                    dimension [B, T, N, 3] (T in sequence length,
                                    N is number of points)
        :rtype torch.Tensor: The scene SDF for each point at each timestep
                             (i.e. the minimum SDF value across the M cuboids
                             at each timestep), has dim [B, T, N]
        """
        assert points.ndim == 4

        # We are going to need to map points in the global frame to the cuboid frame
        # First take the points and make them homogeneous by adding a one to the end

        # points_from_volumes = points[self.nonzero_volumes, :, :]
        assert points.size(0) == self.centers.size(0)
        if torch.all(~self.mask):
            return float("inf") * torch.ones(points.shape[:-1]).type_as(points)

        homog_points = torch.cat(
            (
                points,
                torch.ones((*points.shape[:-1], 1)).type_as(points),
            ),
            dim=3,
        )
        # Next, project these points into their respective cuboid frames
        # Will return [B x M x N x 3]

        points_proj = torch.matmul(
            self.inv_frames[:, :, None, None, :, :],
            homog_points[:, None, :, :, :, None],
        ).squeeze(-1)[:, :, :, :, :3]
        B, M, T, N, _ = points_proj.shape
        assert T == points.size(1)
        assert N == points.size(2)
        masked_points = points_proj[self.mask]

        # The follow computations are adapted from here
        # https://github.com/fogleman/sdf/blob/main/sdf/d3.py
        # Move points to top corner

        distances = torch.abs(masked_points) - (self.dims[self.mask] / 2)[:, None, None, :]
        # This is distance only for points outside the box, all points inside return zero
        # This probably needs to be fixed or else there will be a nan gradient

        outside = torch.linalg.norm(torch.maximum(distances, torch.zeros_like(distances)), dim=-1)
        # This is distance for points inside the box, all others return zero
        inner_max_distance = torch.max(distances, dim=-1).values
        inside = torch.minimum(inner_max_distance, torch.zeros_like(inner_max_distance))
        all_sdfs = float("inf") * torch.ones(B, M, T, N).type_as(points)
        all_sdfs[self.mask] = outside + inside
        return torch.min(all_sdfs, dim=1)[0]


class TorchCylinders:
    """
    A Pytorch representation of a batch of M cylinders (i.e. B elements in the batch,
    M cylinders per element). Any of these cylinders can have zero volume (these
    will be masked out during calculation of the various functions in this
    class, such as sdf).
    """

    def __init__(
        self,
        centers: torch.Tensor,
        radii: torch.Tensor,
        heights: torch.Tensor,
        quaternions: torch.Tensor,
    ):
        """
        :param centers torch.Tensor: Has dim [B, M, 3]
        :param radii torch.Tensor: Has dim [B, M, 1]
        :param heights torch.Tensor: Has dim [B, M, 1]
        :param quaternions torch.Tensor: Has dim [B, M, 4] with quaternions formatted as
                                   (w, x, y, z)
        """
        assert centers.ndim == 3
        assert radii.ndim == 3
        assert heights.ndim == 3
        assert quaternions.ndim == 3

        self.radii = radii
        self.heights = heights
        self.centers = centers

        # It's helpful to ensure the quaternions are normalized
        self.quats = quaternions / torch.linalg.norm(quaternions, dim=2)[:, :, None]
        self._init_frames()
        # Mask for nonzero volumes
        self.mask = ~torch.logical_or(
            torch.isclose(self.radii, torch.zeros(1).type_as(centers)).squeeze(-1),
            torch.isclose(self.heights, torch.zeros(1).type_as(centers)).squeeze(-1),
        )

        self.pose_4x4 = torch.zeros((centers.size(0), centers.size(1), 4, 4)).type_as(centers)
        self.pose_4x4[:, :, 3, 3] = 1
        self.pose_4x4[:, :, :3, :3] = quats_to_rot(self.quats)
        self.pose_4x4[:, :, :3, 3] = self.centers

    def surface_area(self) -> torch.Tensor:
        """
        Calculates the surface area of the cylinders

        :rtype torch.Tensor: A tensor of the surface areas of the cylinders
        """
        area = 2 * np.pi * self.radii * (self.radii + self.heights)
        return area.squeeze(-1)

    def sample_surface(self, num_points, noise=0.0):
        """
        Samples random points on the surface of the cylinder using PyTorch.
        Probabilities are weighed based on the area of each side.
        The implementation is fully vectorized for BxMx... arrays.

        :param num_points: The number of points to sample on the surface
        :param noise: The range of uniform noise to apply to samples

        :return: A random pointcloud sampled from the surface of the cylinder
        """
        B, M, _ = self.radii.shape
        assert noise >= 0, "Noise should be non-negative"

        # Create random angles and compute circle points for each cylinder
        angles = (
            torch.rand(B, M, num_points, device=self.radii.device, dtype=self.radii.dtype)
            * 2
            * np.pi
            - np.pi
        )
        circle_points = torch.stack(
            (torch.cos(angles).type_as(self.radii), torch.sin(angles).type_as(self.radii)), dim=-1
        )

        # Compute surface area for each cylinder
        surface_area = self.surface_area()
        # Choose which surface to sample for each point
        probs = torch.stack(
            [
                np.pi * self.radii.squeeze(-1) ** 2 / surface_area,
                self.heights.squeeze(-1) * 2 * np.pi * self.radii.squeeze(-1) / surface_area,
                np.pi * self.radii.squeeze(-1) ** 2 / surface_area,
            ],
            dim=-1,
        )

        # replace nans with 0
        probs[torch.isnan(probs)] = 0

        # replace any row in the last dim of probs that is all 0s with 1/3
        probs[torch.all(probs == 0, dim=-1)] = 1 / 3

        which_surface = torch.multinomial(probs.view(B * M, -1), num_points, replacement=True).view(
            B, M, num_points
        )

        # Adjust circle points based on the chosen surface
        zero_mask = which_surface == 0
        one_mask = which_surface == 1
        two_mask = which_surface == 2
        zero_pts = (
            torch.rand((B, M, num_points), device=self.radii.device, dtype=self.radii.dtype)
            * self.radii
            * zero_mask
        )
        one_pts = self.radii.repeat(1, 1, num_points) * one_mask
        two_pts = (
            torch.rand((B, M, num_points), device=self.radii.device, dtype=self.radii.dtype)
            * self.radii
            * two_mask
        )

        circle_points = (zero_pts + one_pts + two_pts).unsqueeze(-1) * circle_points

        # Compute z-coordinates
        z = (
            zero_mask * -self.heights / 2
            + one_mask
            * (
                torch.rand((B, M, num_points), device=self.radii.device, dtype=self.radii.dtype)
                * self.heights
                - self.heights / 2
            )
            + two_mask * self.heights / 2
        )

        # Concatenate to get the surface points
        surface_points = torch.cat((circle_points, z.unsqueeze(-1)), dim=-1)

        # Transform points to world coordinates
        surface_points = transform_points(surface_points, self.pose_4x4)
        # Apply noise
        surface_points += torch.rand_like(surface_points) * 2 * noise - noise
        # surface_points *= (1-zero_probs_mask).unsqueeze(-1).unsqueeze(-1)
        surface_points[torch.isnan(surface_points)] = 0
        return surface_points

    def geometrout(self):
        """
        Helper method to convert this into geometrout primitives
        """
        B, M, _ = self.centers.shape
        return [
            [
                Cylinder(
                    center=self.centers[bidx, midx, :].detach().cpu().numpy(),
                    radius=self.radii[bidx, midx, 0].detach().cpu().numpy(),
                    height=self.heights[bidx, midx, 0].detach().cpu().numpy(),
                    quaternion=self.quats[bidx, midx, :].detach().cpu().numpy(),
                )
                for midx in range(M)
                if self.mask[bidx, midx]
            ]
            for bidx in range(B)
        ]

    def _init_frames(self):
        """
        In order to calculate the SDF, we need to calculate the inverse
        transformation of the cylinder. This is because we are transforming points
        in the world frame into the cylinder frame.
        """
        B = self.centers.size(0)
        M = self.centers.size(1)

        R = quats_to_rot(-self.quats)
        t = self.centers

        R_inv = R.transpose(-2, -1)

        t_inv = -torch.matmul(R_inv, t.unsqueeze(-1)).squeeze(-1)

        self.inv_frames = torch.zeros((B, M, 4, 4)).type_as(self.centers)
        self.inv_frames[:, :, :3, :3] = R_inv
        self.inv_frames[:, :, :3, 3] = t_inv
        self.inv_frames[:, :, 3, 3] = 1
        self.inv_frames[torch.isnan(self.inv_frames)] = 0

    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        """
        :param points torch.Tensor: The points with which to calculate the SDF, has
                                    dim [B, N, 3] (N is the number of points)
        :rtype torch.Tensor: The scene SDF value for each point (i.e. the minimum SDF
                             value for each of the M cylinders), has dim [B, N]
        """
        assert points.ndim == 3
        assert points.size(0) == self.centers.size(0)
        if torch.all(~self.mask):
            return float("inf") * torch.ones(points.size(0), points.size(1)).type_as(points)

        homog_points = torch.cat(
            (
                points,
                torch.ones((points.size(0), points.size(1), 1)).type_as(points),
            ),
            dim=2,
        )
        # Next, project these points into their respective cylinder frames
        # Will return [B x M x N x 3]
        points_proj = torch.matmul(
            self.inv_frames[:, :, None, :, :], homog_points[:, None, :, :, None]
        ).squeeze(-1)[:, :, :, :3]
        B, M, N, _ = points_proj.shape
        masked_points = points_proj[self.mask]

        surface_distance_xy = torch.linalg.norm(masked_points[:, :, :2], dim=2)
        z_distance = masked_points[:, :, 2]

        half_extents_2d = torch.stack((self.radii[self.mask], self.heights[self.mask] / 2), dim=2)
        points_2d = torch.stack((surface_distance_xy, z_distance), dim=2)
        distances_2d = torch.abs(points_2d) - half_extents_2d

        # This is distance only for points outside the box, all points inside return zero
        outside = torch.linalg.norm(
            torch.maximum(distances_2d, torch.zeros_like(distances_2d)), dim=2
        )
        # This is distance for points inside the box, all others return zero
        inner_max_distance_2d = torch.max(distances_2d, dim=2).values
        inside = torch.minimum(inner_max_distance_2d, torch.zeros_like(inner_max_distance_2d))

        all_sdfs = float("inf") * torch.ones(B, M, N).type_as(points)
        all_sdfs[self.mask] = outside + inside
        return torch.min(all_sdfs, dim=1)[0]

    def sdf_sequence(self, points: torch.Tensor) -> torch.Tensor:
        """
        Calculates SDF values for a time sequence of point clouds
        :param points torch.Tensor: The batched sequence of point clouds with
                                    dimension [B, T, N, 3] (T in sequence length,
                                    N is number of points)
        :rtype torch.Tensor: The scene SDF for each point at each timestep
                             (i.e. the minimum SDF value across the M cylinders at
                             each timestep), has dim [B, T, N]
        """
        assert points.ndim == 4

        # We are going to need to map points in the global frame to the cylinder frame
        # First take the points and make them homogeneous by adding a one to the end

        assert points.size(0) == self.centers.size(0)
        if torch.all(~self.mask):
            return float("inf") * torch.ones(points.shape[:-1]).type_as(points)

        homog_points = torch.cat(
            (
                points,
                torch.ones((*points.shape[:-1], 1)).type_as(points),
            ),
            dim=3,
        )
        # Next, project these points into their respective cylinder frames
        # Will return [B x M x N x 3]

        points_proj = torch.matmul(
            self.inv_frames[:, :, None, None, :, :],
            homog_points[:, None, :, :, :, None],
        ).squeeze(-1)[:, :, :, :, :3]
        B, M, T, N, _ = points_proj.shape
        assert T == points.size(1)
        assert N == points.size(2)
        masked_points = points_proj[self.mask]

        surface_distance_xy = torch.linalg.norm(masked_points[:, :, :, :2], dim=-1)
        z_distance = masked_points[:, :, :, 2]

        half_extents_2d = torch.stack((self.radii[self.mask], self.heights[self.mask] / 2), dim=2)[
            :, :, None, :
        ]
        points_2d = torch.stack((surface_distance_xy, z_distance), dim=3)
        distances_2d = torch.abs(points_2d) - half_extents_2d

        # This is distance only for points outside the box, all points inside return zero
        outside = torch.linalg.norm(
            torch.maximum(distances_2d, torch.zeros_like(distances_2d)), dim=3
        )
        # This is distance for points inside the box, all others return zero
        inner_max_distance_2d = torch.max(distances_2d, dim=3).values
        inside = torch.minimum(inner_max_distance_2d, torch.zeros_like(inner_max_distance_2d))

        all_sdfs = float("inf") * torch.ones(B, M, T, N).type_as(points)
        all_sdfs[self.mask] = outside + inside
        return torch.min(all_sdfs, dim=1)[0]


class TorchCapsules:
    """
    A Pytorch representation of a batch of M cylinders (i.e. B elements in the batch,
    M capsules per element). This class is mainly used for vectorized point cloud generation
    in IsaacGym.

    Note: in IsaacGym, quaternions are in xyzw format, but TorchCapsules uses wxyz
    """

    def __init__(
        self,
        centers: torch.Tensor,
        radii: torch.Tensor,
        heights: torch.Tensor,
        quaternions: torch.Tensor,
    ):
        """
        :param centers torch.Tensor: Has dim [B, M, 3]
        :param radii torch.Tensor: Has dim [B, M, 1]
        :param heights (distance between two semi-sphere centers) torch.Tensor: Has dim [B, M, 1]
        :param quaternions torch.Tensor: Has dim [B, M, 4] with quaternions formatted as
                                   (w, x, y, z)
        """
        assert centers.ndim == 3
        assert radii.ndim == 3
        assert heights.ndim == 3
        assert quaternions.ndim == 3

        self.radii = radii
        self.heights = heights
        self.centers = centers
        # It's helpful to ensure the quaternions are normalized
        self.quats = quaternions / torch.linalg.norm(quaternions, dim=2)[:, :, None]

        self.pose_4x4 = torch.zeros((centers.size(0), centers.size(1), 4, 4)).type_as(centers)
        self.pose_4x4[:, :, 3, 3] = 1
        self.pose_4x4[:, :, :3, :3] = quats_to_rot(self.quats)
        self.pose_4x4[:, :, :3, 3] = self.centers

    def surface_area(self) -> torch.Tensor:
        """
        Calculates the surface area of the capsules

        :rtype torch.Tensor: A tensor of the surface areas of the capsules
        """
        area = 2 * np.pi * self.radii * self.heights + 4 * np.pi * self.radii**2
        return area.squeeze(-1)

    def sample_surface(self, num_points, noise=0.0):
        """
        Samples random points on the surface of the capsule using PyTorch.
        Probabilities are weighed based on the area of each side.
        The implementation is fully vectorized for BxMx... arrays.

        :param num_points: The number of points to sample on the surface
        :param noise: The range of uniform noise to apply to samples

        :return: A random pointcloud sampled from the surface of the capsule
        """
        B, M, _ = self.radii.shape
        assert noise >= 0, "Noise should be non-negative"

        # Create random angles and compute circle points for each cylinder
        angles = (
            torch.rand(B, M, num_points, device=self.radii.device, dtype=self.radii.dtype)
            * 2
            * np.pi
            - np.pi
        )
        circle_points = torch.stack(
            (torch.cos(angles).type_as(self.radii), torch.sin(angles).type_as(self.radii)), dim=-1
        )

        # Compute surface area for each capsule
        surface_area = self.surface_area()
        # Choose which surface to sample for each point
        probs = torch.stack(
            [
                2 * np.pi * self.radii.squeeze(-1) ** 2 / surface_area,
                self.heights.squeeze(-1) * 2 * np.pi * self.radii.squeeze(-1) / surface_area,
                2 * np.pi * self.radii.squeeze(-1) ** 2 / surface_area,
            ],
            dim=-1,
        )

        # replace nans with 0
        probs[torch.isnan(probs)] = 0

        # replace any row in the last dim of probs that is all 0s with 1/3
        probs[torch.all(probs == 0, dim=-1)] = 1 / 3

        which_surface = torch.multinomial(probs.view(B * M, -1), num_points, replacement=True).view(
            B, M, num_points
        )

        # Adjust circle points based on the chosen surface
        zero_mask = which_surface == 0
        one_mask = which_surface == 1
        two_mask = which_surface == 2
        zero_pts = (
            torch.rand((B, M, num_points), device=self.radii.device, dtype=self.radii.dtype)
            * self.radii
            * zero_mask
        )
        one_pts = self.radii.repeat(1, 1, num_points) * one_mask
        two_pts = (
            torch.rand((B, M, num_points), device=self.radii.device, dtype=self.radii.dtype)
            * self.radii
            * two_mask
        )
        circle_points = (zero_pts + one_pts + two_pts).unsqueeze(-1) * circle_points

        # Compute z-coordinates
        z_sphere = torch.sqrt(
            self.radii**2 - circle_points[:, :, :, 0] ** 2 - circle_points[:, :, :, 1] ** 2
        )
        # due to precision err, the value inside torch.sqrt() might be negative
        z_sphere[torch.isnan(z_sphere)] = 0
        z = (
            zero_mask * (-self.heights / 2 - z_sphere)
            + one_mask
            * (
                torch.rand((B, M, num_points), device=self.radii.device, dtype=self.radii.dtype)
                * self.heights
                - self.heights / 2
            )
            + two_mask * (self.heights / 2 + z_sphere)
        )

        # Concatenate to get the surface points
        surface_points = torch.cat((z.unsqueeze(-1), circle_points), dim=-1)
        # Transform points to world coordinates
        surface_points = transform_points(surface_points, self.pose_4x4)
        # Apply noise
        if noise > 0:
            surface_points += torch.rand_like(surface_points) * 2 * noise - noise
        # surface_points *= (1-zero_probs_mask).unsqueeze(-1).unsqueeze(-1)
        surface_points[torch.isnan(surface_points)] = 0
        return surface_points


class ObjaMesh:
    def __init__(self, position, scale, quaternion, obj_id, mesh_id, max_points=1000):
        meshes_dir = os.path.join(os.path.dirname(__file__), "..", "..", "meshes")
        type_mapping_file_path = os.path.join(meshes_dir, "type_mapping.json")
        obj_int2str = {}
        try:
            with open(type_mapping_file_path, "r") as file:
                obj_str2int = json.load(file)
            obj_int2str = {value: key for key, value in obj_str2int.items()}
        except FileNotFoundError:
            print("The file does not exist.")
        subdirectory = obj_int2str[int(obj_id)]
        self._pose = SE3(xyz=position, so3=SO3(quaternion=quaternion))
        self.scale = scale
        self.quaternion = quaternion
        self.file_path = os.path.abspath(os.path.join(meshes_dir, subdirectory, mesh_id + ".npy"))
        self.pc = np.load(self.file_path)
        self.surface_area = self.scale**2 * 6  # just assume a cube

    def calculate_convex_hull_area_vectorized(self):
        "Quick estimation of SA based on point cloud. Note that this is just a rough estimation"
        hull = ConvexHull(self.pc)
        triangles = self.pc[hull.simplices]
        edges1 = triangles[:, 1] - triangles[:, 0]
        edges2 = triangles[:, 2] - triangles[:, 0]
        cross_products = np.cross(edges1, edges2)
        areas = 0.5 * np.linalg.norm(cross_products, axis=1)

        return np.sum(areas)

    @property
    def position(self):
        """
        :return: The position of the mesh as a list
        """
        return self._pose.xyz

    @position.setter
    def position(self, val):
        """
        Set the position of the mesh
        :param val: The new position of the mesh
        """
        self._pose.xyz = val

    @property
    def pose(self):
        """
        :return: The pose of the mesh as an SE3 object
        """
        return self._pose

    def is_zero_volume(self):
        """
        Check if the mesh is considered to have zero volume.
        """
        return False

    def sample_surface(self, num_points, noise=0.0):
        indices = np.random.choice(self.pc.shape[0], num_points, replace=True)
        sampled_points = self.pc[indices]

        scaled_points = sampled_points * self.scale
        rotation_matrix = self.quaternion_to_rotation_matrix(self.quaternion)
        rotated_points = np.dot(scaled_points, rotation_matrix.T)

        if noise > 0:
            noise_vector = np.random.normal(0, noise, rotated_points.shape)
            rotated_points += noise_vector

        return rotated_points + np.array(self.position)

    def quaternion_to_rotation_matrix(self, q):
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
            ]
        )


def construct_mixed_point_cloud(
    obstacles: Sequence[Union[Sphere, Cuboid, Cylinder, ObjaMesh]], num_points: int
) -> np.ndarray:
    """
    Creates a random point cloud from a collection of obstacles. The points in
    the point cloud should be fairly(-ish) distributed amongst the obstacles based
    on their surface area.

    :param obstacles Sequence[Union[Sphere, Cuboid, Cylinder]]: The obstacles in the scene
    :param num_points int: The total number of points in the samples scene (not
                           the number of points per obstacle)
    :rtype np.ndarray: Has dim [N, 3] where N is num_points
    """
    point_set = []
    total_obstacles = len(obstacles)
    if total_obstacles == 0:
        return np.array([[]])

    # Allocate points based on obstacle surface area for even sampling
    surface_areas = np.array([o.surface_area for o in obstacles])
    total_area = np.sum(surface_areas)
    proportions = (surface_areas / total_area).tolist()

    indices = list(range(1, total_obstacles + 1))
    random.shuffle(indices)
    idx = 0

    for o, prop in zip(obstacles, proportions):
        sample_number = int(prop * num_points) + 500
        samples = o.sample_surface(sample_number)
        _points = indices[idx] * np.ones((sample_number, 4))
        _points[:, :3] = samples
        point_set.append(_points)
        idx += 1
    points = np.concatenate(point_set, axis=0)

    # Downsample to the desired number of points
    return points[np.random.choice(points.shape[0], num_points, replace=False), :]


def generate_shifted_mask(proportions, num_points, shift, ind0, ind1, ind2):
    """
    Generates a mask for a given set of proportions and shift values. The mask
    is a binary array of shape [B, M, N] where B is the batch size, M is the
    number of obstacles, and N is the number of points.
    The mask has the property that the first nonzero element in each row is at
    shift[b][m].
    """
    B, M = proportions.shape
    N = num_points

    # Create a range tensor that has the same shape as the final desired output
    range_tensor = torch.arange(N, device=proportions.device).expand(B, M, N)

    # Expand props to match the dimensions of range_tensor
    expanded_props = proportions.unsqueeze(-1).expand_as(range_tensor)

    # Create a mask where elements of range_tensor are less than expanded_props
    mask = range_tensor >= expanded_props

    # Use this mask to create the total tensor
    mask_arr = torch.ones((B, M, N), device=proportions.device)
    mask_arr[mask] = 0

    # shift the mask so that the first nonzero element is at shift[b][m]
    # total shape: [B, M, N]
    # shift shape: [B, M]
    rolled_ind2 = torch.remainder(ind2 - shift[:, :, None], N)
    mask_arr = mask_arr[ind0, ind1, rolled_ind2]
    return mask_arr


# @torch.compile
def construct_mixed_point_cloud_torch(
    cuboid_centers,
    cuboid_dims,
    cuboid_quaternions,
    cylinder_centers,
    cylinder_radii,
    cylinder_heights,
    cylinder_quaternions,
    sphere_centers,
    sphere_radii,
    num_points: int,
    num_extra_points: int,
    ind0,
    ind1,
    ind2,
) -> np.ndarray:
    """
    Creates a random point cloud from a collection of obstacles. The points in
    the point cloud should be fairly(-ish) distributed amongst the obstacles based
    on their surface area.

    :param cuboid_centers torch.Tensor: Has dim [B, M, 3]
    :param cuboid_dims torch.Tensor: Has dim [B, M, 3]
    :param cuboid_quaternions torch.Tensor: Has dim [B, M, 4] with quaternions formatted as
                                      (w, x, y, z)
    :param cylinder_centers torch.Tensor: Has dim [B, M, 3]
    :param cylinder_radii torch.Tensor: Has dim [B, M, 1]
    :param cylinder_heights torch.Tensor: Has dim [B, M, 1]
    :param cylinder_quaternions torch.Tensor: Has dim [B, M, 4] with quaternions formatted as
                                        (w, x, y, z)
    :param sphere_centers torch.Tensor: Has dim [B, M, 3]
    :param sphere_radii torch.Tensor: Has dim [B, M, 1]
    :param num_points int: The total number of points in the samples scene (not
                           the number of points per obstacle)

    :rtype np.ndarray: Has dim [B, N, 3] where N is num_points
    """

    # Creating instances of the obstacles
    spheres = TorchSpheres(sphere_centers, sphere_radii)
    cylinders = TorchCylinders(
        cylinder_centers, cylinder_radii, cylinder_heights, cylinder_quaternions
    )
    cuboids = TorchCuboids(cuboid_centers, cuboid_dims, cuboid_quaternions)

    # Allocate points based on obstacle surface area for even sampling
    surface_areas = torch.cat(
        (spheres.surface_area(), cylinders.surface_area(), cuboids.surface_area()), dim=1
    )  # [B, M*3]
    total_area = surface_areas.sum(dim=1).unsqueeze(-1)  # [B]
    proportions = ((surface_areas / total_area) * num_points).int() + num_extra_points  # [B, M*3]
    num_points = 3 * num_extra_points + num_points  # add 500 points to each component
    proportions = (
        proportions * (surface_areas > 0).int()
    )  # [B, M*3] mask out the components that are not present
    sphere_proportions, cylinder_proportions, cuboid_proportions = torch.chunk(
        proportions, 3, dim=1
    )  # [B, M]

    # mask of shape [B, M, N] in which the number of True for each b, m is sphere_proportions[b, m]
    sphere_mask = generate_shifted_mask(
        sphere_proportions, num_points, torch.zeros_like(sphere_proportions), ind0, ind1, ind2
    )
    spheres_samples = spheres.sample_surface(num_points)  # [B, M, N, 3]
    sphere_masked_samples = spheres_samples * sphere_mask.unsqueeze(-1)  # [B, M, N, 3]

    cylinder_mask = generate_shifted_mask(
        cylinder_proportions, num_points, sphere_proportions, ind0, ind1, ind2
    )
    cylinders_samples = cylinders.sample_surface(num_points)
    cylinder_masked_samples = cylinders_samples * cylinder_mask.unsqueeze(-1)

    cuboid_mask = generate_shifted_mask(
        cuboid_proportions, num_points, sphere_proportions + cylinder_proportions, ind0, ind1, ind2
    )
    cuboids_samples = cuboids.sample_surface(num_points)
    cuboid_masked_samples = cuboids_samples * cuboid_mask.unsqueeze(-1)

    points = sphere_masked_samples + cylinder_masked_samples + cuboid_masked_samples
    points = points.reshape(points.size(0), -1, 3)  # B, M, N, 3 -> B, M*N, 3
    # shuffle the pointclouds
    points = points.index_select(1, torch.randperm(points.shape[1], device=points.device))
    # put all the zero points at the end
    not_zero_mask = torch.all(points != 0, dim=-1).int()
    sorted_indices = torch.argsort(not_zero_mask, dim=1, descending=True)
    points = points[torch.arange(points.size(0)).unsqueeze(-1), sorted_indices, :]
    return points


def construct_mixed_point_cloud_ig(
    gp_dim,
    gp_center,
    gp_quaternion,
    cuboid_dims,
    cuboid_centers,
    cuboid_quaternions,
    capsule_radii,
    capsule_heights,
    capsule_centers,
    capsule_quaternions,
    sphere_centers,
    sphere_radii,
    num_points: int,
    num_extra_points: int,
    num_gp_points: int = 200,
) -> np.ndarray:
    """
    Creates a random point cloud from a collection of obstacles. The points in
    the point cloud should be fairly(-ish) distributed amongst the obstacles based
    on their surface area. Note: in isaacgym, the number of obstacle of each kind is the same

    :param cuboid_centers torch.Tensor: Has dim [B, M, 3]
    :param cuboid_dims torch.Tensor: Has dim [B, M, 3]
    :param cuboid_quaternions torch.Tensor: Has dim [B, M, 4] with quaternions formatted as
                                      (w, x, y, z)
    :param capsule_centers torch.Tensor: Has dim [B, M, 3]
    :param capsule_radii torch.Tensor: Has dim [B, M, 1]
    :param capsule_heights torch.Tensor: Has dim [B, M, 1]
    :param capsule_quaternions torch.Tensor: Has dim [B, M, 4] with quaternions formatted as
                                        (w, x, y, z)
    :param sphere_centers torch.Tensor: Has dim [B, M, 3]
    :param sphere_radii torch.Tensor: Has dim [B, M, 1]
    :param num_points int: The total number of points in the samples scene (not
                           the number of points per obstacle)
    :param num_extra_points int: The number of extra points to add to each obstacle

    :rtype np.ndarray: Has dim [B, N, 3] where N is num_points
    """
    B, M = cuboid_centers.shape[:2]
    num_obs_points = num_points - num_gp_points
    total_obs_points = 3 * num_extra_points + num_obs_points  # add extra points to each component

    # Creating instances of the obstacles
    spheres = TorchSpheres(sphere_centers, sphere_radii)
    capsules = TorchCapsules(capsule_centers, capsule_radii, capsule_heights, capsule_quaternions)
    cuboids = TorchCuboids(cuboid_centers, cuboid_dims, cuboid_quaternions)
    ground_plane = TorchCuboids(gp_center, gp_dim, gp_quaternion)

    # Allocate points based on obstacle surface area for even sampling
    surface_areas = torch.cat(
        (spheres.surface_area(), capsules.surface_area(), cuboids.surface_area()), dim=1
    )  # [B, M*3]
    total_area = surface_areas.sum(dim=1).unsqueeze(-1)  # [B]
    proportions = (
        (surface_areas / total_area) * num_obs_points
    ).int() + num_extra_points  # [B, M*3]
    proportions = (
        proportions * (surface_areas > 0).int()
    )  # [B, M*3] mask out the components that are not present

    sphere_proportions, capsule_proportions, cuboid_proportions = torch.chunk(
        proportions, 3, dim=1
    )  # [B, M]

    # mask of shape [B, M, N] in which the number of True for each b, m is sphere_proportions[b, m]
    ind0 = torch.arange(B, device=proportions.device)[:, None, None].expand(B, M, total_obs_points)
    ind1 = torch.arange(M, device=proportions.device)[None, :, None].expand(B, M, total_obs_points)
    ind2 = torch.arange(total_obs_points, device=proportions.device)[None, None, :].expand(
        B, M, total_obs_points
    )
    sphere_mask = generate_shifted_mask(
        sphere_proportions,
        total_obs_points,
        torch.zeros_like(sphere_proportions),
        ind0,
        ind1,
        ind2,
    )
    spheres_samples = spheres.sample_surface(total_obs_points)  # [B, M, N, 3]
    sphere_masked_samples = spheres_samples * sphere_mask.unsqueeze(-1)  # [B, M, N, 3]

    capsule_mask = generate_shifted_mask(
        capsule_proportions,
        total_obs_points,
        sphere_proportions,
        ind0,
        ind1,
        ind2,
    )
    capsules_samples = capsules.sample_surface(total_obs_points)
    capsule_masked_samples = capsules_samples * capsule_mask.unsqueeze(-1)

    cuboid_mask = generate_shifted_mask(
        cuboid_proportions,
        total_obs_points,
        sphere_proportions + capsule_proportions,
        ind0,
        ind1,
        ind2,
    )
    cuboids_samples = cuboids.sample_surface(total_obs_points)
    cuboid_masked_samples = cuboids_samples * cuboid_mask.unsqueeze(-1)

    gp_samples = ground_plane.sample_surface(num_gp_points).reshape(
        B, -1, 3
    )  # B, 1, N,3 -> B, N, 3
    obs_points = sphere_masked_samples + capsule_masked_samples + cuboid_masked_samples
    obs_points = obs_points.reshape(B, -1, 3)  # B, M, N, 3 -> B, M*N, 3
    # shuffle the pointclouds
    obs_points = obs_points.index_select(
        1, torch.randperm(obs_points.shape[1], device=obs_points.device)
    )
    # put all the zero points at the end, TODO get rid of this/rewrite to be much faster
    not_zero_mask = torch.any(obs_points != 0, dim=-1).int()
    not_underground_mask = (obs_points[..., 2] >= 0).int()
    sorted_indices = torch.argsort(not_zero_mask * not_underground_mask, dim=1, descending=True)
    obs_points = obs_points[torch.arange(B).unsqueeze(-1), sorted_indices, :]
    obs_points = obs_points[:, :num_obs_points, :]
    points = torch.cat((obs_points, gp_samples), dim=1)
    return points[:, :num_points, :]


if __name__ == "__main__":
    import torch

    torch.manual_seed(1)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    B, M, N = 256, 15, 4096
    N = N // M
    num_extra_points = 100
    device = torch.device("cuda")

    cuboid_centers = torch.randn((B, M, 3), device=device).float()
    cuboid_dims = torch.ones((B, M, 3), device=device).float()
    cuboid_quaternions = torch.tensor([1, 0, 0, 0], device=device).repeat(B, M, 1).float()

    cylinder_centers = torch.randn((B, M, 3), device=device).float()
    cylinder_radii = torch.ones((B, M, 1), device=device).float()
    cylinder_heights = torch.ones((B, M, 1), device=device).float()
    cylinder_quaternions = torch.tensor([1, 0, 0, 0], device=device).repeat(B, M, 1).float()

    sphere_radii = torch.ones((B, M, 1), device=device).float()
    sphere_centers = torch.randn((B, M, 3), device=device).float()

    N_modified = N + num_extra_points * 3
    ind0 = torch.arange(B, device=device)[:, None, None].expand(B, M, N_modified)
    ind1 = torch.arange(M, device=device)[None, :, None].expand(B, M, N_modified)
    ind2 = torch.arange(N_modified, device=device)[None, None, :].expand(B, M, N_modified)
    pts = construct_mixed_point_cloud_torch(
        cuboid_centers,
        cuboid_dims,
        cuboid_quaternions,
        cylinder_centers,
        cylinder_radii,
        cylinder_heights,
        cylinder_quaternions,
        sphere_centers,
        sphere_radii,
        N,
        num_extra_points,
        ind0,
        ind1,
        ind2,
    )
    import time

    s0 = time.time()

    pts = construct_mixed_point_cloud_torch(
        cuboid_centers,
        cuboid_dims,
        cuboid_quaternions,
        cylinder_centers,
        cylinder_radii,
        cylinder_heights,
        cylinder_quaternions,
        sphere_centers,
        sphere_radii,
        N,
        num_extra_points,
        ind0,
        ind1,
        ind2,
    )
    print(time.time() - s0)
    print(pts.shape)

    # fk_sampler = FrankaSampler(
    #     "cpu",
    #     use_cache=False,
    #     default_prismatic_value=0.025,
    # )

    # save pcd as ply
    # import open3d as o3d

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pts[0].cpu().numpy())
    # o3d.io.write_point_cloud("test.ply", pcd)

    # img = render_single_pointcloud(pts[0])
    # cv2.imwrite("test.png", img)

    # import time
    # joint_angles = torch.randn((B, 7), device=device).float()
    # robot_pts = fk_sampler.sample(joint_angles, num_points=2048)
    # s0 = time.time()
    # robot_pts = fk_sampler.sample(joint_angles, num_points=2048)
    # print(robot_pts.shape)
    # print(time.time() - s0)
