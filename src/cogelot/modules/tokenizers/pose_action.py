from typing import cast

import numpy as np
import torch
from numpy import typing as npt

from cogelot.structures.token import PoseActionToken
from cogelot.structures.vima import (
    N_DISCRETE_ROT_BINS,
    N_DISCRETE_X_BINS,
    N_DISCRETE_Y_BINS,
    ROT_MAX,
    ROT_MIN,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    PoseAction,
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


def convert_discrete_action_to_continuous_on_boundary(
    discrete_actions: torch.Tensor, boundary: torch.Tensor, mask_for_select: torch.Tensor
) -> torch.Tensor:
    """Convert discrete actions to continuous actions on the boundary."""
    # Make sure the discrete actions have the same number of dimensions as the mask for select
    if discrete_actions.ndim != mask_for_select.ndim:
        dim_difference = abs(discrete_actions.ndim - mask_for_select.ndim)
        if discrete_actions.ndim > mask_for_select.ndim:
            for _ in range(dim_difference):
                mask_for_select = mask_for_select.unsqueeze(-1)
        if mask_for_select.ndim > discrete_actions.ndim:
            for _ in range(dim_difference):
                discrete_actions = discrete_actions.unsqueeze(-1)

    # As the mask is likely just covering the number of timesteps, we need to expand it to cover
    # all the axes
    mask_for_select = mask_for_select.expand_as(discrete_actions)

    # Index the actions on the boundaries to get the continuous value for each axis. We use masked
    # select to ignore the actions that are set to -100, which is the ignore index.
    source = boundary[torch.masked_select(discrete_actions, mask_for_select)]

    tensor = torch.masked_scatter(
        input=discrete_actions.to(source.dtype), mask=mask_for_select, source=source
    )
    return tensor


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
        rot_boundary_min: float = ROT_MIN,
        rot_boundary_max: float = ROT_MAX,
        n_discrete_rot_bins: int = N_DISCRETE_ROT_BINS,
    ) -> None:
        self._x_boundaries = torch.linspace(
            start=x_boundary_min,
            end=x_boundary_max,
            steps=n_discrete_x_bins,
        )
        self._y_boundaries = torch.linspace(
            start=y_boundary_min, end=y_boundary_max, steps=n_discrete_y_bins
        )
        self._rot_boundaries = torch.linspace(
            start=rot_boundary_min,
            end=rot_boundary_max,
            steps=n_discrete_rot_bins,
        )

    def convert_continuous_to_discrete(
        self, actions: dict[PoseActionType, torch.Tensor]
    ) -> dict[PoseActionType, torch.Tensor]:
        """Convert continuous actions to discrete tokens."""
        x_boundary, y_boundary, rot_boundary = self._get_boundaries(
            actions["pose0_position"].device
        )

        discrete_actions: dict[PoseActionType, torch.Tensor] = {
            k: v.clone().detach() for k, v in actions.items()
        }

        discrete_actions["pose0_position"][..., 0] = torch.bucketize(
            discrete_actions["pose0_position"][..., 0].contiguous(), x_boundary
        )
        discrete_actions["pose0_position"][..., 1] = torch.bucketize(
            discrete_actions["pose0_position"][..., 1].contiguous(), y_boundary
        )
        discrete_actions["pose0_rotation"] = torch.bucketize(
            discrete_actions["pose0_rotation"].contiguous(), rot_boundary
        )

        discrete_actions["pose1_position"][..., 0] = torch.bucketize(
            discrete_actions["pose1_position"][..., 0].contiguous(), x_boundary
        )
        discrete_actions["pose1_position"][..., 1] = torch.bucketize(
            discrete_actions["pose1_position"][..., 1].contiguous(), y_boundary
        )
        discrete_actions["pose1_rotation"] = torch.bucketize(
            discrete_actions["pose1_rotation"].contiguous(), rot_boundary
        )
        discrete_actions = {k: v.long() for k, v in discrete_actions.items()}
        return discrete_actions

    def convert_discrete_to_continuous(
        self,
        actions: dict[PoseActionType, torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> dict[PoseActionType, torch.Tensor]:
        """Convert discrete actions to continuous actions."""
        # If the mask is None, then we can just set all the values to False
        if mask is None:
            mask = torch.zeros(
                actions["pose0_position"].shape[:-1],
                dtype=torch.bool,
                device=actions["pose0_position"].device,
            )

        # Make sure the actions are integers so we can use them as indices
        actions = {k: v.long() for k, v in actions.items()}

        # Create the boundaries
        x_boundary, y_boundary, rot_boundary = self._get_boundaries(
            actions["pose0_position"].device
        )
        continuous_actions: dict[PoseActionType, torch.Tensor] = {
            k: v.clone().float() for k, v in actions.items()
        }

        # Since the mask assumes that the True values are to be removed, we want to invert it for
        # the masked selecting since we want to keep all the un-masked values
        mask_for_select = ~mask

        # Index the actions on the boundaries to get the continuous value for each axis. We use
        # masked scatter because some action tokens are set to -100, which is the ignore index.
        continuous_actions["pose0_position"][..., 0] = (
            convert_discrete_action_to_continuous_on_boundary(
                actions["pose0_position"][..., 0], x_boundary, mask_for_select
            )
        )
        continuous_actions["pose0_position"][..., 1] = (
            convert_discrete_action_to_continuous_on_boundary(
                actions["pose0_position"][..., 1], y_boundary, mask_for_select
            )
        )
        continuous_actions["pose0_rotation"] = convert_discrete_action_to_continuous_on_boundary(
            actions["pose0_rotation"], rot_boundary, mask_for_select
        )
        continuous_actions["pose1_position"][..., 0] = (
            convert_discrete_action_to_continuous_on_boundary(
                actions["pose1_position"][..., 0], x_boundary, mask_for_select
            )
        )
        continuous_actions["pose1_position"][..., 1] = (
            convert_discrete_action_to_continuous_on_boundary(
                actions["pose1_position"][..., 1], y_boundary, mask_for_select
            )
        )
        continuous_actions["pose1_rotation"] = convert_discrete_action_to_continuous_on_boundary(
            actions["pose1_rotation"], rot_boundary, mask_for_select
        )

        return continuous_actions

    def tokenize(self, actions: list[PoseAction]) -> list[PoseActionToken]:
        """Tokenize actions into discrete actions."""
        # When dumping the model, we are explicitly excluding the index from the dump because that
        # is something that should not get passed to the tokenizer
        discrete_actions = (
            self.convert_continuous_to_discrete(
                cast(dict[PoseActionType, torch.Tensor], action.model_dump(exclude={"index"}))
            )
            for action in actions
        )
        indexed_discrete_actions = (
            (action.index, discrete_action)
            for action, discrete_action in zip(actions, discrete_actions, strict=True)
        )
        tokens = [
            PoseActionToken.model_validate({"index": idx, **action})
            for idx, action in indexed_discrete_actions
        ]
        return tokens

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get all the boundaries, and clone and move them to another device."""
        x_boundary = self._x_boundaries.clone().to(device)
        y_boundary = self._y_boundaries.clone().to(device)
        rot_boundary = self._rot_boundaries.clone().to(device)
        return x_boundary, y_boundary, rot_boundary
