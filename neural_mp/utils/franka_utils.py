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

from typing import Tuple, Union

import numpy as np
import torch

from neural_mp.utils.constants import (
    FRANKA_ACCELERATION_LIMIT,
    FRANKA_LOWER_LIMITS,
    FRANKA_UPPER_LIMITS,
    FRANKA_VELOCITY_LIMIT,
)

JOINT_LIMITS = np.array([FRANKA_LOWER_LIMITS, FRANKA_UPPER_LIMITS]).transpose()


def _normalize_franka_joints_numpy(
    batch_trajectory: np.ndarray,
    limits: Tuple[float, float] = (-1, 1),
    use_real_constraints: bool = True,
) -> np.ndarray:
    """
    Normalizes joint angles to be within a specified range according to the Franka's
    joint limits. This is the numpy version

    :param batch_trajectory np.ndarray: A batch of trajectories. Can have dims
                                        [7] if a single configuration
                                        [B, 7] if a batch of configurations
                                        [B, T, 7] if a batched time-series of configurations
    :param limits Tuple[float, float]: The new limits to map to
    :param use_real_constraints bool: If true, use the empirically determined joint limits
                                      (this is unpublished--just found by monkeying around
                                      with the robot).
                                      If false, use the published joint limits from Franka
    :rtype np.ndarray: An array with the same dimensions as the input
    """
    franka_limits = JOINT_LIMITS
    normalized = (batch_trajectory - franka_limits[:, 0]) / (
        franka_limits[:, 1] - franka_limits[:, 0]
    ) * (limits[1] - limits[0]) + limits[0]
    return normalized


def _normalize_franka_joints_torch(
    batch_trajectory: torch.Tensor,
    limits: Tuple[float, float] = (-1, 1),
    use_real_constraints: bool = True,
) -> torch.Tensor:
    """
    Normalizes joint angles to be within a specified range according to the Franka's
    joint limits. This is the torch version

    :param batch_trajectory torch.Tensor: A batch of trajectories. Can have dims
                                        [7] if a single configuration
                                        [B, 7] if a batch of configurations
                                        [B, T, 7] if a batched time-series of configurations
    :param limits Tuple[float, float]: The new limits to map to
    :param use_real_constraints bool: If true, use the empirically determined joint limits
                                      (this is unpublished--just found by monkeying around
                                      with the robot).
                                      If false, use the published joint limits from Franka
    :rtype torch.Tensor: A tensor with the same dimensions as the input
    """
    assert isinstance(batch_trajectory, torch.Tensor)
    franka_limits = torch.as_tensor(JOINT_LIMITS).type_as(batch_trajectory)
    return (batch_trajectory - franka_limits[:, 0]) / (
        franka_limits[:, 1] - franka_limits[:, 0]
    ) * (limits[1] - limits[0]) + limits[0]


def normalize_franka_joints(
    batch_trajectory: Union[np.ndarray, torch.Tensor],
    limits: Tuple[float, float] = (-1, 1),
    use_real_constraints: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalizes joint angles to be within a specified range according to the Franka's
    joint limits. This is semantic sugar that dispatches to the correct implementation.

    :param batch_trajectory Union[np.ndarray, torch.Tensor]: A batch of trajectories. Can have dims
                                        [7] if a single configuration
                                        [B, 7] if a batch of configurations
                                        [B, T, 7] if a batched time-series of configurations
    :param limits Tuple[float, float]: The new limits to map to
    :param use_real_constraints bool: If true, use the empirically determined joint limits
                                      (this is unpublished--just found by monkeying around
                                      with the robot).
                                      If false, use the published joint limits from Franka
    :rtype Union[np.ndarray, torch.Tensor]: A tensor or numpy array with the same dimensions
                                            and type as the input
    :raises NotImplementedError: Raises an error if another data type (e.g. a list) is passed in
    """
    if isinstance(batch_trajectory, torch.Tensor):
        return _normalize_franka_joints_torch(
            batch_trajectory, limits=limits, use_real_constraints=True
        )
    elif isinstance(batch_trajectory, np.ndarray):
        return _normalize_franka_joints_numpy(
            batch_trajectory, limits=limits, use_real_constraints=True
        )
    else:
        raise NotImplementedError("Only torch.Tensor and np.ndarray implemented")


def _unnormalize_franka_joints_numpy(
    batch_trajectory: np.ndarray,
    limits: Tuple[float, float] = (-1, 1),
    use_real_constraints: bool = True,
) -> np.ndarray:
    """
    Unnormalizes joint angles from a specified range back into the Franka's joint limits.
    This is the numpy version and the inverse of `_normalize_franka_joints_numpy`.

    :param batch_trajectory np.ndarray: A batch of trajectories. Can have dims
                                        [7] if a single configuration
                                        [B, 7] if a batch of configurations
                                        [B, T, 7] if a batched time-series of configurations
    :param limits Tuple[float, float]: The current limits to map to the joint limits
    :param use_real_constraints bool: If true, use the empirically determined joint limits
                                      (this is unpublished--just found by monkeying around
                                      with the robot).
                                      If false, use the published joint limits from Franka
    :rtype np.ndarray: An array with the same dimensions as the input
    """
    franka_limits = JOINT_LIMITS
    assert np.all(batch_trajectory >= limits[0])
    assert np.all(batch_trajectory <= limits[1])
    franka_limit_range = franka_limits[:, 1] - franka_limits[:, 0]
    franka_lower_limit = franka_limits[:, 0]
    for _ in range(batch_trajectory.ndim - 1):
        franka_limit_range = franka_limit_range[np.newaxis, ...]
        franka_lower_limit = franka_lower_limit[np.newaxis, ...]
    unnormalized = (batch_trajectory - limits[0]) * franka_limit_range / (
        limits[1] - limits[0]
    ) + franka_lower_limit

    return unnormalized


def _unnormalize_franka_joints_torch(
    batch_trajectory: torch.Tensor,
    limits: Tuple[float, float] = (-1, 1),
    use_real_constraints: bool = True,
) -> torch.Tensor:
    """
    Unnormalizes joint angles from a specified range back into the Franka's joint limits.
    This is the torch version and the inverse of `_normalize_franka_joints_torch`.

    :param batch_trajectory torch.Tensor: A batch of trajectories. Can have dims
                                        [7] if a single configuration
                                        [B, 7] if a batch of configurations
                                        [B, T, 7] if a batched time-series of configurations
    :param limits Tuple[float, float]: The current limits to map to the joint limits
    :param use_real_constraints bool: If true, use the empirically determined joint limits
                                      (this is unpublished--just found by monkeying around
                                      with the robot).
                                      If false, use the published joint limits from Franka
    :rtype torch.Tensor: A tensor with the same dimensions as the input
    """
    assert isinstance(batch_trajectory, torch.Tensor)
    franka_limits = torch.as_tensor(JOINT_LIMITS).type_as(batch_trajectory)
    assert torch.all(batch_trajectory >= limits[0])
    assert torch.all(batch_trajectory <= limits[1]), batch_trajectory.max()
    franka_limit_range = franka_limits[:, 1] - franka_limits[:, 0]
    franka_lower_limit = franka_limits[:, 0]
    for _ in range(batch_trajectory.ndim - 1):
        franka_limit_range = franka_limit_range.unsqueeze(0)
        franka_lower_limit = franka_lower_limit.unsqueeze(0)
    return (batch_trajectory - limits[0]) * franka_limit_range / (
        limits[1] - limits[0]
    ) + franka_lower_limit


def unnormalize_franka_joints(
    batch_trajectory: Union[np.ndarray, torch.Tensor],
    limits: Tuple[float, float] = (-1, 1),
    use_real_constraints: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Unnormalizes joint angles from a specified range back into the Franka's joint limits.
    This is semantic sugar that dispatches to the correct implementation, the inverse of
    `normalize_franka_joints`.

    :param batch_trajectory Union[np.ndarray, torch.Tensor]: A batch of trajectories. Can have dims
                                        [7] if a single configuration
                                        [B, 7] if a batch of configurations
                                        [B, T, 7] if a batched time-series of configurations
    :param limits Tuple[float, float]: The current limits to map to the joint limits
    :param use_real_constraints bool: If true, use the empirically determined joint limits
                                      (this is unpublished--just found by monkeying around
                                      with the robot).
                                      If false, use the published joint limits from Franka
    :rtype Union[np.ndarray, torch.Tensor]: A tensor or numpy array with the same dimensions
                                            and type as the input
    :raises NotImplementedError: Raises an error if another data type (e.g. a list) is passed in
    """
    if isinstance(batch_trajectory, torch.Tensor):
        return _unnormalize_franka_joints_torch(
            batch_trajectory, limits=limits, use_real_constraints=use_real_constraints
        )
    elif isinstance(batch_trajectory, np.ndarray):
        return _unnormalize_franka_joints_numpy(
            batch_trajectory, limits=limits, use_real_constraints=use_real_constraints
        )
    else:
        raise NotImplementedError("Only torch.Tensor and np.ndarray implemented")
