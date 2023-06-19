import torch

from cogelot.structures.token import PoseActionToken
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
        discrete_actions = (
            (
                self.discretize_actions(action.to_tensor())
                if action.is_continuous
                else action.to_tensor()
            )
            for action in actions
        )
        indexed_discrete_actions = (
            (action.index, discrete_action)
            for action, discrete_action in zip(actions, discrete_actions, strict=True)
        )

        # Convert to tokens
        tokens = [
            PoseActionToken.parse_obj({"index": idx, **action})
            for idx, action in indexed_discrete_actions
        ]

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

    def discretize_actions(
        self, actions: dict[PoseActionType, torch.Tensor]
    ) -> dict[PoseActionType, torch.Tensor]:
        """Convert continuous actions into discrete actions."""
        device = actions["pose0_position"].device
        boundary_x = torch.linspace(start=0, end=1, steps=self._n_discrete_x_bins, device=device)
        boundary_y = torch.linspace(start=0, end=1, steps=self._n_discrete_y_bins, device=device)
        boundary_rot = torch.linspace(
            start=0, end=1, steps=self._n_discrete_rotation_bins, device=device
        )

        actions["pose0_position"][..., 0] = torch.bucketize(
            actions["pose0_position"][..., 0].contiguous(), boundary_x
        )
        actions["pose0_position"][..., 1] = torch.bucketize(
            actions["pose0_position"][..., 1].contiguous(), boundary_y
        )
        actions["pose0_rotation"] = torch.bucketize(
            actions["pose0_rotation"].contiguous(), boundary_rot
        )

        actions["pose1_position"][..., 0] = torch.bucketize(
            actions["pose1_position"][..., 0].contiguous(), boundary_x
        )
        actions["pose1_position"][..., 1] = torch.bucketize(
            actions["pose1_position"][..., 1].contiguous(), boundary_y
        )
        actions["pose1_rotation"] = torch.bucketize(
            actions["pose1_rotation"].contiguous(), boundary_rot
        )
        actions = {k: v.long() for k, v in actions.items()}
        return actions
