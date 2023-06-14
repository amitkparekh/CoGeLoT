from __future__ import annotations

from typing import TYPE_CHECKING

from cogelot.structures.token import PoseActionToken


if TYPE_CHECKING:
    import torch

    from cogelot.structures.vima import PoseAction, PoseActionType


class PoseActionTokenizer:
    """Tokenize actions into discrete actions for the encoding."""

    def __init__(
        self,
        n_discrete_x_bins: int,
        n_discrete_y_bins: int,
        n_discrete_z_bins: int,
        n_discrete_rotation_bins: int,
    ) -> None:
        self._n_discrete_x_bins = n_discrete_x_bins
        self._n_discrete_y_bins = n_discrete_y_bins
        self._n_discrete_z_bins = n_discrete_z_bins
        self._n_discrete_rotation_bins = n_discrete_rotation_bins

    def tokenize(self, actions: list[PoseAction]) -> list[PoseActionToken]:
        """Tokenize actions into discrete actions."""
        # Convert to tensors
        actions_as_tensors = (action.to_tensor() for action in actions)
        # Make continuous
        continuous_actions = (self.de_discretize_actions(action) for action in actions_as_tensors)
        # Convert to tokens
        tokens = [PoseActionToken.parse_obj(action) for action in continuous_actions]

        return tokens

    def de_discretize_actions(
        self, actions: dict[PoseActionType, torch.Tensor]
    ) -> dict[PoseActionType, torch.Tensor]:
        """Convert discrete actions into continuous actions."""
        actions = {k: v.float() for k, v in actions.items()}
        actions["pose0_position"][..., 0] = (
            actions["pose0_position"][..., 0] / self._n_discrete_x_bins
        )
        actions["pose0_position"][..., 1] = (
            actions["pose0_position"][..., 1] / self._n_discrete_y_bins
        )
        actions["pose0_rotation"] = actions["pose0_rotation"] / self._n_discrete_rotation_bins

        actions["pose1_position"][..., 0] = (
            actions["pose1_position"][..., 0] / self._n_discrete_x_bins
        )
        actions["pose1_position"][..., 1] = (
            actions["pose1_position"][..., 1] / self._n_discrete_y_bins
        )
        actions["pose1_rotation"] = actions["pose1_rotation"] / self._n_discrete_rotation_bins
        return actions
