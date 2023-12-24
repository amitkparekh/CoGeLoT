from typing import cast

import numpy as np
import torch
from einops import rearrange
from numpy import typing as npt

from cogelot.structures.vima import (
    N_DISCRETE_ROT_BINS,
    N_DISCRETE_X_BINS,
    N_DISCRETE_Y_BINS,
    N_DISCRETE_Z_BINS,
    ROT_MAX,
    ROT_MIN,
    STARTING_POSITION_ENV,
    STARTING_ROTATION,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    Z_MAX,
    Z_MIN,
    PoseActionType,
)
from vima.utils import DataDict


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
        remove_z_position_dim: bool = False,
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

        self._remove_z_position_dim = remove_z_position_dim

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
        actions = self._clamp_discrete_tokens_to_limits(actions)
        actions = {k: v.float() for k, v in actions.items()}

        actions = self._convert_discrete_to_rescaled_continuous(actions)
        actions = self._restore_rescaled_continuous_to_correct_range(actions)

        return actions

    def convert_discrete_token_to_environment(
        self,
        action_token: dict[PoseActionType, torch.Tensor],
        *,
        should_remove_zth_position_dim: bool = True,
    ) -> dict[PoseActionType, npt.NDArray[np.float32]]:
        """Convert discrete pose aciton tokens to the environment."""
        actions = self.convert_discrete_to_continuous(action_token)

        if should_remove_zth_position_dim:
            actions["pose0_position"] = actions["pose0_position"][:2]
            actions["pose1_position"] = actions["pose1_position"][:2]

        # Convert to numpy because it needs to be in numpy for the environment
        actions_numpy = {k: v.cpu().numpy() for k, v in actions.items()}
        # actions_numpy = any_slice(actions_numpy, np.s_[0, 0])

        return cast(dict[PoseActionType, npt.NDArray[np.float32]], actions_numpy)

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
        z_boundary = torch.linspace(
            start=0,
            end=1,
            steps=self._n_discrete_z_bins,
            device=device,
        )
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
        subtract_from_position = self._position_minimum.to(device)
        divide_from_position = self._position_range.to(device)

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
        actions["pose0_rotation"] = torch.bucketize(
            actions["pose0_rotation"].contiguous(), rot_boundary
        )
        actions["pose1_position"][..., 0] = torch.bucketize(
            actions["pose1_position"][..., 0].contiguous(), x_boundary
        )
        actions["pose1_position"][..., 1] = torch.bucketize(
            actions["pose1_position"][..., 1].contiguous(), y_boundary
        )
        actions["pose1_rotation"] = torch.bucketize(
            actions["pose1_rotation"].contiguous(), rot_boundary
        )
        if not self._remove_z_position_dim:
            actions["pose0_position"][..., 2] = torch.bucketize(
                actions["pose0_position"][..., 2].contiguous(), z_boundary
            )
            actions["pose1_position"][..., 2] = torch.bucketize(
                actions["pose1_position"][..., 2].contiguous(), z_boundary
            )
        actions = {k: v.long() for k, v in actions.items()}
        return actions

    def _convert_discrete_to_rescaled_continuous(
        self, actions: dict[PoseActionType, torch.Tensor]
    ) -> dict[PoseActionType, torch.Tensor]:
        """Convert the discrete values to rescaled continuous values."""
        device = actions["pose0_position"].device
        divide_from_position = self._position_bins.to(device)

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
        add_to_position = self._position_minimum.to(device)
        multiply_to_position = self._position_range.to(device)

        actions["pose0_position"].multiply_(multiply_to_position).add_(add_to_position)
        actions["pose1_position"].multiply_(multiply_to_position).add_(add_to_position)

        actions["pose0_rotation"].multiply_(self._rot_boundary_max - self._rot_boundary_min).add_(
            self._rot_boundary_min
        )
        actions["pose1_rotation"].multiply_(self._rot_boundary_max - self._rot_boundary_min).add_(
            self._rot_boundary_min
        )

        return actions

    def _clamp_discrete_tokens_to_limits(
        self, discrete_actions: dict[PoseActionType, torch.Tensor]
    ) -> dict[PoseActionType, torch.Tensor]:
        """Clamp the discrete tokens to the limits of the axis.

        Just in case the model predicts a value that is out of range.
        """
        device = discrete_actions["pose0_position"].device
        position_max = self._position_bins.to(device)
        discrete_actions["pose0_position"].clamp_(max=position_max)
        discrete_actions["pose1_position"].clamp_(max=position_max)
        discrete_actions["pose0_rotation"].clamp_(max=self._n_discrete_rot_bins)
        discrete_actions["pose1_rotation"].clamp_(max=self._n_discrete_rot_bins)
        return discrete_actions

    def _clamp_continuous_actions_to_limits(
        self, continuous_actions: dict[PoseActionType, torch.Tensor]
    ) -> dict[PoseActionType, torch.Tensor]:
        """Clamp continuous coordinates to the limits of the environment."""
        device = continuous_actions["pose0_position"].device
        position_min = self._position_minimum.to(device)
        position_max = self._position_maximum.to(device)
        continuous_actions["pose0_position"].clamp_(min=position_min, max=position_max)
        continuous_actions["pose1_position"].clamp_(min=position_min, max=position_max)
        continuous_actions["pose0_rotation"].clamp_(
            min=self._rot_boundary_min, max=self._rot_boundary_max
        )
        continuous_actions["pose1_rotation"].clamp_(
            min=self._rot_boundary_min, max=self._rot_boundary_max
        )

        return continuous_actions

    @property
    def _position_minimum(self) -> torch.Tensor:
        """Get a tensor of minimum positions."""
        return torch.tensor(
            [
                self._x_boundary_min,
                self._y_boundary_min,
                self._z_boundary_min,
            ][: self._num_position_dims]
        )

    @property
    def _position_maximum(self) -> torch.Tensor:
        """Get a tensor of maximum positions."""
        return torch.tensor(
            [
                self._x_boundary_max,
                self._y_boundary_max,
                self._z_boundary_max,
            ][: self._num_position_dims]
        )

    @property
    def _position_range(self) -> torch.Tensor:
        """Get a tensor of position ranges."""
        return torch.tensor(
            [
                self._x_boundary_max - self._x_boundary_min,
                self._y_boundary_max - self._y_boundary_min,
                self._z_boundary_max - self._z_boundary_min,
            ][: self._num_position_dims],
        )

    @property
    def _position_bins(self) -> torch.Tensor:
        """Get a tensor of position bins."""
        return torch.tensor(
            [
                self._n_discrete_x_bins,
                self._n_discrete_y_bins,
                self._n_discrete_z_bins,
            ][: self._num_position_dims],
        )

    @property
    def _num_position_dims(self) -> int:
        """Get the number of position dimensions.

        This changes depending on whether the z position dimension needs to be removed or not.
        """
        return 2 if self._remove_z_position_dim else 3


@torch.no_grad()
def prepare_target_actions(
    continuous_targart_actions: dict[PoseActionType, torch.Tensor],
    pose_action_tokenizer: PoseActionTokenizer,
    *,
    ignore_target_index: int = -100,
) -> torch.Tensor:
    """Convert the discrete target actions into a single tensor."""
    discrete_target_actions = pose_action_tokenizer.convert_continuous_to_discrete(
        continuous_targart_actions
    )
    # Shape: (batch size, num observations, num action tokens per timestep)
    target_actions_tensor = torch.cat(list(continuous_targart_actions.values()), dim=-1)
    discrete_target_actions_tensor = torch.cat(list(discrete_target_actions.values()), dim=-1)
    discrete_target_actions_tensor[
        target_actions_tensor == ignore_target_index
    ] = ignore_target_index

    # Shape: (num action tokens per pose, batch size, num observations)
    discrete_target_actions_tensor = rearrange(discrete_target_actions_tensor, "B T A -> A B T")
    return discrete_target_actions_tensor


def is_action_pointless(action_for_env: dict[PoseActionType, npt.NDArray[np.float32]]) -> bool:
    """Figure out if the action is pointless, and therefore is the end-of-trajectory action.

    If the robot performs a pointless action, it means that there are no other actions that it
    feels that it should be doing. In this case, we that this as the end of the trajectory action.
    """
    is_pose0_position_pointless = np.allclose(
        action_for_env["pose0_position"], STARTING_POSITION_ENV
    )
    is_pose0_rotation_pointless = np.allclose(action_for_env["pose0_rotation"], STARTING_ROTATION)
    is_pose1_position_pointless = np.allclose(
        action_for_env["pose1_position"], STARTING_POSITION_ENV
    )
    is_pose1_rotation_pointless = np.allclose(action_for_env["pose1_rotation"], STARTING_ROTATION)

    return (
        is_pose0_position_pointless
        and is_pose0_rotation_pointless
        and is_pose1_position_pointless
        and is_pose1_rotation_pointless
    )
