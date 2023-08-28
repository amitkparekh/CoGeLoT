from typing import cast

import numpy as np
import torch
from numpy import typing as npt

from cogelot.structures.token import PoseActionToken
from cogelot.structures.vima import PoseAction, PoseActionType
from vima.utils import any_slice


N_DISCRETE_X_BINS: int = 50
N_DISCRETE_Y_BINS: int = 100
N_DISCRETE_Z_BINS: int = 50
N_DISCRETE_ROT_BINS: int = 50

# Action space boundaries
X_MIN = 0.25
X_MAX = 0.75
Y_MIN = -0.5
Y_MAX = 0.5
Z_MIN = 0
Z_MAX = 0.32
ROT_MIN = -1
ROT_MAX = 1


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
        self, actions: dict[PoseActionType, torch.Tensor]
    ) -> dict[PoseActionType, torch.Tensor]:
        """Convert discrete actions to continuous actions."""
        # Make sure the actions are integers so we can use them as indices
        actions = {k: v.long() for k, v in actions.items()}

        # Create the boundaries
        x_boundary, y_boundary, rot_boundary = self._get_boundaries(
            actions["pose0_position"].device
        )
        continuous_actions: dict[PoseActionType, torch.Tensor] = {
            k: v.clone().float() for k, v in actions.items()
        }

        # Index the actions on the boundaries
        continuous_actions["pose0_position"][..., 0] = x_boundary[
            actions["pose0_position"][..., 0]
        ]
        continuous_actions["pose0_position"][..., 1] = y_boundary[
            actions["pose0_position"][..., 1]
        ]
        continuous_actions["pose0_rotation"] = rot_boundary[actions["pose0_rotation"]]
        continuous_actions["pose1_position"][..., 0] = x_boundary[
            actions["pose1_position"][..., 0]
        ]
        continuous_actions["pose1_position"][..., 1] = y_boundary[
            actions["pose1_position"][..., 1]
        ]
        continuous_actions["pose1_rotation"] = rot_boundary[actions["pose1_rotation"]]

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
