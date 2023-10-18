from typing import cast

import numpy as np
import torch
from numpy import typing as npt

from cogelot.structures.vima import (
    N_DISCRETE_ROT_BINS,
    N_DISCRETE_X_BINS,
    N_DISCRETE_Y_BINS,
    N_DISCRETE_Z_BINS,
    ROT_MAX,
    ROT_MIN,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    Z_MAX,
    Z_MIN,
    PoseActionType,
)
from vima.utils import DataDict, any_slice


def create_mask_from_target_actions(
    actions: DataDict | dict[PoseActionType, torch.Tensor], *, ignore_target_index: int = -100
) -> torch.Tensor:
    """Create a mask from the target actions.

    This is used to mask out the loss for the actions that are not present in the target actions.
    """
    actions = cast(dict[PoseActionType, torch.Tensor], actions)

    # 1. Figure out which axis has a target value that should be masked (i.e. the target value is
    #    -100)
    masked_axes_per_pose = [action.eq(ignore_target_index) for action in actions.values()]

    # 2. Sum across the axes dimension to get a tensor of shape [batch, timesteps]. However, to
    #    make sure that only mask timesteps where _all_ axes are masked, we check that the sum is
    #    equal to the number of axes. If the sum is equal to the number of axes, then all axes are
    #    masked, and are converted to `True`.
    masked_timesteps_per_pose = [
        masked_pose_axes.sum(dim=-1).eq(masked_pose_axes.size(-1))
        for masked_pose_axes in masked_axes_per_pose
    ]

    # 3. Each tensor how has the same shape. We merge all the tensors by summing them together. To
    #    make sure that only timesteps where _all_ poses are masked, we check that the sum is equal
    #    to the number of poses. If the sum is equal to the number of poses, then all poses are
    #    masked, and are converted to `True`.
    masked_timesteps = cast(torch.Tensor, sum(masked_timesteps_per_pose)).eq(
        len(masked_timesteps_per_pose)
    )

    return masked_timesteps


class PoseActionTokenizer:
    """Tokenize actions into discrete actions for the encoding."""

    def __init__(
        self,
        *,
        x_boundary_min: float = X_MIN,
        x_boundary_max: float = X_MAX,
        n_discrete_x_bins: int = N_DISCRETE_X_BINS,
        y_boundary_min: float = Y_MIN,
        y_boundary_max: float = Y_MAX,
        n_discrete_y_bins: int = N_DISCRETE_Y_BINS,
        z_boundary_min: float = Z_MIN,
        z_boundary_max: float = Z_MAX,
        n_discrete_z_bins: int = N_DISCRETE_Z_BINS,
        rot_boundary_min: float = ROT_MIN,
        rot_boundary_max: float = ROT_MAX,
        n_discrete_rot_bins: int = N_DISCRETE_ROT_BINS,
    ) -> None:
        self._n_discrete_x_bins = n_discrete_x_bins
        self._n_discrete_y_bins = n_discrete_y_bins
        self._n_discrete_z_bins = n_discrete_z_bins
        self._n_discrete_rot_bins = n_discrete_rot_bins

        self._x_boundary_min = x_boundary_min
        self._x_boundary_max = x_boundary_max

        self._y_boundary_min = y_boundary_min
        self._y_boundary_max = y_boundary_max

        self._z_boundary_min = z_boundary_min
        self._z_boundary_max = z_boundary_max

        self._rot_boundary_min = rot_boundary_min
        self._rot_boundary_max = rot_boundary_max

    def normalize_continuous_actions(
        self, actions: dict[PoseActionType, torch.Tensor]
    ) -> dict[PoseActionType, torch.Tensor]:
        """Normalise the continuous actions between 0 and 1."""
        actions = {k: v.detach().clone().float() for k, v in actions.items()}
        actions = self._rescale_continuous_actions_to_0_and_1(actions)
        return actions

    def convert_continuous_to_discrete(
        self, actions: dict[PoseActionType, torch.Tensor]
    ) -> dict[PoseActionType, torch.Tensor]:
        """Convert continuous actions to discrete tokens."""
        actions = {k: v.detach().clone() for k, v in actions.items()}
        actions = self._rescale_continuous_actions_to_0_and_1(actions)
        actions = self._convert_rescaled_continuous_to_discrete(actions)
        return actions

    def convert_discrete_to_continuous(
        self, actions: dict[PoseActionType, torch.Tensor]
    ) -> dict[PoseActionType, torch.Tensor]:
        """Convert discrete actions to continuous actions."""
        # Make all of the actions floats
        actions = {k: v.float() for k, v in actions.items()}

        actions = self._convert_discrete_to_rescaled_continuous(actions)
        actions = self._restore_rescaled_continuous_to_correct_range(actions)

        return actions

    def convert_token_to_environment(
        self, action_token: dict[PoseActionType, torch.Tensor]
    ) -> dict[PoseActionType, npt.NDArray[np.float64]]:
        """Convert discrete pose aciton tokens to the environment."""
        actions = self.convert_discrete_to_continuous(action_token)

        # Convert to numpy because it needs to be in numpy for the environment
        actions_numpy = {k: v.cpu().numpy() for k, v in actions.items()}
        actions_numpy = any_slice(actions_numpy, np.s_[0, 0])

        return cast(dict[PoseActionType, npt.NDArray[np.float64]], actions_numpy)

    def _get_boundaries(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get all the boundaries, and clone and move them to another device."""
        x_boundary = torch.linspace(
            start=0,
            end=1,
            steps=self._n_discrete_x_bins,
            device=device,
        )
        y_boundary = torch.linspace(
            start=0,
            end=1,
            steps=self._n_discrete_y_bins,
            device=device,
        )
        z_boundary = torch.linspace(start=0, end=1, steps=self._n_discrete_z_bins, device=device)
        rot_boundary = torch.linspace(
            start=0,
            end=1,
            steps=self._n_discrete_rot_bins,
            device=device,
        )

        return x_boundary, y_boundary, z_boundary, rot_boundary

    def _rescale_continuous_actions_to_0_and_1(
        self, actions: dict[PoseActionType, torch.Tensor]
    ) -> dict[PoseActionType, torch.Tensor]:
        """Rescale all the continuous actions to be between 0 and 1."""
        device = actions["pose0_position"].device

        subtract_from_position = torch.tensor(
            [self._x_boundary_min, self._y_boundary_min, self._z_boundary_min], device=device
        )
        divide_from_position = torch.tensor(
            [
                self._x_boundary_max - self._x_boundary_min,
                self._y_boundary_max - self._y_boundary_min,
                self._z_boundary_max - self._z_boundary_min,
            ]
        )

        actions["pose0_position"].subtract_(subtract_from_position).divide_(divide_from_position)
        actions["pose1_position"].subtract_(subtract_from_position).divide_(divide_from_position)

        actions["pose0_rotation"].subtract_(self._rot_boundary_min).divide_(
            self._rot_boundary_max - self._rot_boundary_min
        )
        actions["pose1_rotation"].subtract_(self._rot_boundary_min).divide_(
            self._rot_boundary_max - self._rot_boundary_min
        )

        return actions

    def _convert_rescaled_continuous_to_discrete(
        self, actions: dict[PoseActionType, torch.Tensor]
    ) -> dict[PoseActionType, torch.Tensor]:
        """Convert the rescaled continuous values to discrete values for tokens."""
        x_boundary, y_boundary, z_boundary, rot_boundary = self._get_boundaries(
            actions["pose0_position"].device
        )

        actions["pose0_position"][..., 0] = torch.bucketize(
            actions["pose0_position"][..., 0].contiguous(), x_boundary
        )
        actions["pose0_position"][..., 1] = torch.bucketize(
            actions["pose0_position"][..., 1].contiguous(), y_boundary
        )
        actions["pose0_position"][..., 2] = torch.bucketize(
            actions["pose0_position"][..., 2].contiguous(), z_boundary
        )
        actions["pose0_rotation"] = torch.bucketize(
            actions["pose0_rotation"].contiguous(), rot_boundary
        )
        actions["pose1_position"][..., 0] = torch.bucketize(
            actions["pose1_position"][..., 0].contiguous(), x_boundary
        )
        actions["pose1_position"][..., 1] = torch.bucketize(
            actions["pose1_position"][..., 1].contiguous(), y_boundary
        )
        actions["pose1_position"][..., 2] = torch.bucketize(
            actions["pose1_position"][..., 2].contiguous(), z_boundary
        )
        actions["pose1_rotation"] = torch.bucketize(
            actions["pose1_rotation"].contiguous(), rot_boundary
        )
        actions = {k: v.long() for k, v in actions.items()}
        return actions

    def _convert_discrete_to_rescaled_continuous(
        self, actions: dict[PoseActionType, torch.Tensor]
    ) -> dict[PoseActionType, torch.Tensor]:
        """Convert the discrete values to rescaled continuous values."""
        device = actions["pose0_position"].device
        divide_from_position = torch.tensor(
            [self._n_discrete_x_bins, self._n_discrete_y_bins, self._n_discrete_z_bins],
            device=device,
        )

        actions["pose0_position"].divide_(divide_from_position)
        actions["pose1_position"].divide_(divide_from_position)
        actions["pose0_rotation"].divide_(self._n_discrete_rot_bins)
        actions["pose1_rotation"].divide_(self._n_discrete_rot_bins)

        return actions

    def _restore_rescaled_continuous_to_correct_range(
        self, actions: dict[PoseActionType, torch.Tensor]
    ) -> dict[PoseActionType, torch.Tensor]:
        """Restore the rescaled continuous values to the correct range."""
        device = actions["pose0_position"].device
        add_to_position = torch.tensor(
            [self._x_boundary_min, self._y_boundary_min, self._z_boundary_min], device=device
        )
        multiply_to_position = torch.tensor(
            [
                self._x_boundary_max - self._x_boundary_min,
                self._y_boundary_max - self._y_boundary_min,
                self._z_boundary_max - self._z_boundary_min,
            ]
        )

        actions["pose0_position"].multiply_(multiply_to_position).add_(add_to_position)
        actions["pose1_position"].multiply_(multiply_to_position).add_(add_to_position)

        actions["pose0_rotation"].multiply_(self._rot_boundary_max - self._rot_boundary_min).add_(
            self._rot_boundary_min
        )
        actions["pose1_rotation"].multiply_(self._rot_boundary_max - self._rot_boundary_min).add_(
            self._rot_boundary_min
        )

        return actions
