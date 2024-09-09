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

from typing import Tuple

import torch
import torch.nn.functional as F

from neural_mp.utils import franka_utils as utils
from neural_mp.utils.geometry import TorchCuboids, TorchCylinders, TorchSpheres
from robofin.pointcloud.torch import FrankaSampler


def point_match_loss(input_pc: torch.Tensor, target_pc: torch.Tensor) -> torch.Tensor:
    """
    A combination L1 and L2 loss to penalize large and small deviations between
    two point clouds

    :param input_pc torch.Tensor: Point cloud sampled from the network's output.
                                  Has dim [B, N, 3]
    :param target_pc torch.Tensor: Point cloud sampled from the supervision
                                   Has dim [B, N, 3]
    :rtype torch.Tensor: The single loss value
    """
    return F.mse_loss(input_pc, target_pc, reduction="mean") + F.l1_loss(
        input_pc, target_pc, reduction="mean"
    )


def collision_loss(
    input_pc: torch.Tensor,
    cuboid_centers: torch.Tensor,
    cuboid_dims: torch.Tensor,
    cuboid_quaternions: torch.Tensor,
    cylinder_centers: torch.Tensor,
    cylinder_radii: torch.Tensor,
    cylinder_heights: torch.Tensor,
    cylinder_quaternions: torch.Tensor,
    sphere_centers: torch.Tensor,
    sphere_radii: torch.Tensor,
    reduction: str = "mean",
    hinge_loss: bool = True,
    margin: float = 0.03,
    smooth_sdf_loss="none",
    compute_loss_on_penetrations_only=True,
) -> torch.Tensor:
    """
    Calculates the hinge loss, calculating whether the robot (represented as a
    point cloud) is in collision with any obstacles in the scene. Collision
    here actually means within 3cm of the obstacle--this is to provide stronger
    gradient signal to encourage the robot to move out of the way. Also, some of the
    primitives can have zero volume (i.e. a dim is zero for cuboids or radius or height is zero for cylinders).
    If these are zero volume, they will have infinite sdf values (and therefore be ignored by the loss).

    :param input_pc torch.Tensor: Points sampled from the robot's surface after it
                                  is placed at the network's output prediction. Has dim [B, N, 3]
    :param cuboid_centers torch.Tensor: Has dim [B, M1, 3]
    :param cuboid_dims torch.Tensor: Has dim [B, M1, 3]
    :param cuboid_quaternions torch.Tensor: Has dim [B, M1, 4]. Quaternion is formatted as w, x, y, z.
    :param cylinder_centers torch.Tensor: Has dim [B, M2, 3]
    :param cylinder_radii torch.Tensor: Has dim [B, M2, 1]
    :param cylinder_heights torch.Tensor: Has dim [B, M2, 1]
    :param cylinder_quaternions torch.Tensor: Has dim [B, M2, 4]. Quaternion is formatted as w, x, y, z.
    :rtype torch.Tensor: Returns the loss value aggregated over the batch
    """

    cuboids = TorchCuboids(
        cuboid_centers,
        cuboid_dims,
        cuboid_quaternions,
    )
    cylinders = TorchCylinders(
        cylinder_centers,
        cylinder_radii,
        cylinder_heights,
        cylinder_quaternions,
    )
    spheres = TorchSpheres(
        sphere_centers,
        sphere_radii,
    )
    # take min of sdf values
    sdf_values = torch.min(
        torch.min(
            cuboids.sdf(input_pc),
            cylinders.sdf(input_pc),
        ),
        spheres.sdf(input_pc),
    )
    sdf_collision = (torch.min(sdf_values, dim=1).values <= 0.0).float()
    if hinge_loss:
        loss = F.hinge_embedding_loss(
            sdf_values,
            -torch.ones_like(sdf_values),
            margin=margin,
            reduction=reduction,
        )
    else:
        # sdf values are > 0 outside of the object, < 0 inside the object
        if compute_loss_on_penetrations_only:
            sdf_values = torch.clamp(
                sdf_values, max=0.0
            )  # only keep the negative values (penetrations)
            num_nonzero = torch.sum(sdf_values != 0.0, dim=1)
            # make sure to not divide by zero
            num_nonzero = torch.clamp(num_nonzero, min=1.0)
            sdf_values = (
                torch.sum(sdf_values, dim=1) / num_nonzero
            )  # if sdf_values is all zeros, then this will be zero

            # do the same thing for the mean across the batch, ie discard zeros (no penetrations at all) when computing the mean
            num_nonzero_batch = torch.sum(sdf_values != 0.0)

            # make sure to not divide by zero
            num_nonzero_batch = torch.clamp(num_nonzero_batch, min=1.0)
            sdf_values *= -1.0  # reverse the sdf because we want to minimize it
            # loss is 0 or positive if inside the object (which we want to minimize)
            if reduction == "mean":
                loss = torch.sum(sdf_values) / num_nonzero_batch
            elif reduction == "sum":
                loss = torch.sum(sdf_values)
            elif reduction == "max":
                loss = torch.max(sdf_values)
        else:
            if smooth_sdf_loss != "none":
                if smooth_sdf_loss == "log_one_plus_exp":
                    loss = torch.log(1 + torch.exp(-sdf_values))
                elif smooth_sdf_loss == "exp":
                    loss = torch.exp(-sdf_values)
            else:
                loss = -sdf_values
            if reduction == "mean":
                loss = torch.mean(loss)
            elif reduction == "sum":
                loss = torch.sum(loss, dim=1).mean()
            elif reduction == "max":
                loss = torch.max(loss, dim=1).values.mean()
    return loss, sdf_collision


class CollisionAndBCLossContainer:
    """
    A container class to hold the various losses. This is structured as a
    container because that allows it to cache the robot pointcloud sampler
    object. By caching this, we reduce parsing time when processing the URDF
    and allow for a consistent random pointcloud (consistent per-GPU, that is)
    """

    def __init__(
        self,
    ):
        self.fk_sampler = None
        self.num_points = 1024

    def __call__(
        self,
        input_normalized: torch.Tensor,
        cuboid_centers: torch.Tensor,
        cuboid_dims: torch.Tensor,
        cuboid_quaternions: torch.Tensor,
        cylinder_centers: torch.Tensor,
        cylinder_radii: torch.Tensor,
        cylinder_heights: torch.Tensor,
        cylinder_quaternions: torch.Tensor,
        sphere_centers: torch.Tensor,
        sphere_radii: torch.Tensor,
        target_normalized: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method calculates both constituent loss function after loading,
        and then caching, a fixed robot point cloud sampler (i.e. the task
        spaces sampled are always the same, as opposed to a random point cloud).
        The fixed point cloud is important for loss calculation so that
        it's possible to take mse between the two pointclouds.

        :param input_normalized torch.Tensor: Has dim [B, 7] and is always between -1 and 1
        :param cuboid_centers torch.Tensor: Has dim [B, M1, 3]
        :param cuboid_dims torch.Tensor: Has dim [B, M1, 3]
        :param cuboid_quaternions torch.Tensor: Has dim [B, M1, 4]. Quaternion is formatted as w, x, y, z.
        :param cylinder_centers torch.Tensor: Has dim [B, M2, 3]
        :param cylinder_radii torch.Tensor: Has dim [B, M2, 1]
        :param cylinder_heights torch.Tensor: Has dim [B, M2, 1]
        :param cylinder_quaternions torch.Tensor: Has dim [B, M2, 4]. Quaternion is formatted as w, x, y, z.
        :param target_normalized torch.Tensor: Has dim [B, 7] and is always between -1 and 1
        :rtype Tuple[torch.Tensor, torch.Tensor]: The two losses aggregated over the batch
        """
        if self.fk_sampler is None:
            self.fk_sampler = FrankaSampler(
                input_normalized.device,
                num_fixed_points=self.num_points,
                use_cache=True,
                default_prismatic_value=0.025,
            )
        input_unnormalized = utils.unnormalize_franka_joints(input_normalized)
        target_unnormalized = utils.unnormalize_franka_joints(target_normalized)
        input_pc = self.fk_sampler.sample(
            input_unnormalized,
        )
        target_pc = self.fk_sampler.sample(
            target_unnormalized,
        )
        return (
            collision_loss(
                input_pc,
                cuboid_centers,
                cuboid_dims,
                cuboid_quaternions,
                cylinder_centers,
                cylinder_radii,
                cylinder_heights,
                cylinder_quaternions,
                sphere_centers,
                sphere_radii,
            ),
            point_match_loss(input_pc, target_pc),
        )


class CollisionLossContainer:
    """
    A container class to hold the various losses. This is structured as a
    container because that allows it to cache the robot pointcloud sampler
    object. By caching this, we reduce parsing time when processing the URDF
    and allow for a consistent random pointcloud (consistent per-GPU, that is)
    """

    def __init__(
        self,
    ):
        self.fk_sampler = None
        self.num_points = 4096

    def __call__(
        self,
        input_unnormalized: torch.Tensor,
        cuboid_centers: torch.Tensor,
        cuboid_dims: torch.Tensor,
        cuboid_quaternions: torch.Tensor,
        cylinder_centers: torch.Tensor,
        cylinder_radii: torch.Tensor,
        cylinder_heights: torch.Tensor,
        cylinder_quaternions: torch.Tensor,
        sphere_centers: torch.Tensor,
        sphere_radii: torch.Tensor,
        reduction: str = "mean",
        hinge_loss: bool = True,
        margin: float = 0.03,
        smooth_sdf_loss="none",
        compute_loss_on_penetrations_only=True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method calculates both constituent loss function after loading,
        and then caching, a fixed robot point cloud sampler (i.e. the task
        spaces sampled are always the same, as opposed to a random point cloud).
        The fixed point cloud is important for loss calculation so that
        it's possible to take mse between the two pointclouds.

        :param input_unnormalized torch.Tensor: Has dim [B, 7]
        :param cuboid_centers torch.Tensor: Has dim [B, M1, 3]
        :param cuboid_dims torch.Tensor: Has dim [B, M1, 3]
        :param cuboid_quaternions torch.Tensor: Has dim [B, M1, 4]. Quaternion is formatted as w, x, y, z.
        :param cylinder_centers torch.Tensor: Has dim [B, M2, 3]
        :param cylinder_radii torch.Tensor: Has dim [B, M2, 1]
        :param cylinder_heights torch.Tensor: Has dim [B, M2, 1]
        :param cylinder_quaternions torch.Tensor: Has dim [B, M2, 4]. Quaternion is formatted as w, x, y, z.
        :param target_normalized torch.Tensor: Has dim [B, 7] and is always between -1 and 1
        """
        if self.fk_sampler is None:
            self.fk_sampler = FrankaSampler(
                input_unnormalized.device,
                num_fixed_points=self.num_points,
                use_cache=True,
                default_prismatic_value=0.025,
            )
        input_pc = self.fk_sampler.sample(
            input_unnormalized,
        )
        return collision_loss(
            input_pc,
            cuboid_centers,
            cuboid_dims,
            cuboid_quaternions,
            cylinder_centers,
            cylinder_radii,
            cylinder_heights,
            cylinder_quaternions,
            sphere_centers,
            sphere_radii,
            reduction,
            hinge_loss,
            margin,
            smooth_sdf_loss,
            compute_loss_on_penetrations_only,
        )
