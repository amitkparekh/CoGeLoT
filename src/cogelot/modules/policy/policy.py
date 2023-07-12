from abc import ABC, abstractmethod

import torch

from cogelot.structures.model import RawPromptTokenType
from cogelot.structures.vima import PoseActionType
from vima.nn.action_decoder.dists import MultiCategorical
from vima.utils import DataDict


class Policy(ABC, torch.nn.Module):
    """Base class for policies.

    This inherits straight from `torch.nn.Module`, so you can super to it.
    """

    n_discrete_x_bins: int = 50
    n_discrete_y_bins: int = 100
    n_discrete_z_bins: int = 50
    n_discrete_rot_bins: int = 50

    def forward(
        self,
        encoded_prompt: torch.Tensor,
        encoded_prompt_mask: torch.Tensor,
        embedded_observations: torch.Tensor,
        embedded_observations_mask: torch.Tensor,
        embedded_actions: torch.Tensor | None,
    ) -> torch.Tensor:
        """Predict the action token."""
        return self.predict_action_token(
            encoded_prompt=encoded_prompt,
            encoded_prompt_mask=encoded_prompt_mask,
            embedded_observations=embedded_observations,
            embedded_observations_mask=embedded_observations_mask,
            embedded_actions=embedded_actions,
        )

    @abstractmethod
    def assemble_prompt(
        self, prompts: tuple[RawPromptTokenType, torch.Tensor, DataDict]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble and embed prompts."""
        ...  # noqa: WPS428

    @abstractmethod
    def embed_observation_token(self, observation: DataDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed observations for an instance."""
        ...  # noqa: WPS428

    @abstractmethod
    def embed_action_token(self, actions: DataDict) -> torch.Tensor:
        """Embed the actions into a tensor."""
        ...  # noqa: WPS428

    @abstractmethod
    def encode_prompt(
        self, embedded_prompt: torch.Tensor, embedded_prompt_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode the prompt."""
        ...  # noqa: WPS428

    @abstractmethod
    def predict_action_token(
        self,
        encoded_prompt: torch.Tensor,
        encoded_prompt_mask: torch.Tensor,
        embedded_observations: torch.Tensor,
        embedded_observations_mask: torch.Tensor,
        embedded_actions: torch.Tensor | None,
    ) -> torch.Tensor:
        """Predict the action token."""
        ...  # noqa: WPS428

    @abstractmethod
    def decode_action_token(
        self, predicted_action_tokens: torch.Tensor
    ) -> dict[PoseActionType, MultiCategorical]:
        """Decode the action token."""
        ...  # noqa: WPS428

    def discretize_action(
        self, action: dict[PoseActionType, torch.Tensor]
    ) -> dict[PoseActionType, torch.Tensor]:
        """Discretize the action."""
        device = action["pose0_position"].device
        boundary_x = torch.linspace(start=0, end=1, steps=self.n_discrete_x_bins, device=device)
        boundary_y = torch.linspace(start=0, end=1, steps=self.n_discrete_y_bins, device=device)
        boundary_rot = torch.linspace(
            start=0, end=1, steps=self.n_discrete_rot_bins, device=device
        )

        action["pose0_position"][..., 0] = torch.bucketize(
            action["pose0_position"][..., 0].contiguous(), boundary_x
        )
        action["pose0_position"][..., 1] = torch.bucketize(
            action["pose0_position"][..., 1].contiguous(), boundary_y
        )
        action["pose0_rotation"] = torch.bucketize(
            action["pose0_rotation"].contiguous(), boundary_rot
        )

        action["pose1_position"][..., 0] = torch.bucketize(
            action["pose1_position"][..., 0].contiguous(), boundary_x
        )
        action["pose1_position"][..., 1] = torch.bucketize(
            action["pose1_position"][..., 1].contiguous(), boundary_y
        )
        action["pose1_rotation"] = torch.bucketize(
            action["pose1_rotation"].contiguous(), boundary_rot
        )
        action = {k: v.long() for k, v in action.items()}
        return action

    def de_discretize_actions(
        self, actions: dict[PoseActionType, torch.Tensor]
    ) -> dict[PoseActionType, torch.Tensor]:
        """De-discretize the actions."""
        actions = {k: v.float() for k, v in actions.items()}
        actions["pose0_position"][..., 0] = (
            actions["pose0_position"][..., 0] / self.n_discrete_x_bins
        )
        actions["pose0_position"][..., 1] = (
            actions["pose0_position"][..., 1] / self.n_discrete_y_bins
        )
        actions["pose0_rotation"] = actions["pose0_rotation"] / self.n_discrete_rot_bins

        actions["pose1_position"][..., 0] = (
            actions["pose1_position"][..., 0] / self.n_discrete_x_bins
        )
        actions["pose1_position"][..., 1] = (
            actions["pose1_position"][..., 1] / self.n_discrete_y_bins
        )
        actions["pose1_rotation"] = actions["pose1_rotation"] / self.n_discrete_rot_bins
        return actions
