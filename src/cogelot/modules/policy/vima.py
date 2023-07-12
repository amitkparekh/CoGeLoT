from typing import cast

import torch

from cogelot.modules.policy.policy import Policy
from cogelot.structures.model import RawPromptTokenType
from cogelot.structures.vima import PoseActionType
from vima.nn.action_decoder.dists import MultiCategorical
from vima.policy import VIMAPolicy as TheirPolicy
from vima.utils import DataDict


class VIMAPolicy(Policy, TheirPolicy):
    """Wrap their policy in our interface."""

    def assemble_prompt(
        self, prompts: tuple[RawPromptTokenType, torch.Tensor, DataDict]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble and embed the prompt."""
        return TheirPolicy.forward_prompt_assembly(self, prompts)

    def encode_prompt(
        self, embedded_prompt: torch.Tensor, embedded_prompt_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode the prompt."""
        return TheirPolicy.forward_prepared_prompt(self, embedded_prompt, embedded_prompt_mask)

    def embed_observation_token(self, observation: DataDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed an observation."""
        return cast(
            tuple[torch.Tensor, torch.Tensor], TheirPolicy.forward_obs_token(self, observation)
        )

    def embed_action_token(self, actions: DataDict) -> torch.Tensor:
        """Embed the actions into a tensor."""
        return TheirPolicy.forward_action_token(self, actions)

    def predict_action_token(
        self,
        encoded_prompt: torch.Tensor,
        encoded_prompt_mask: torch.Tensor,
        embedded_observations: torch.Tensor,
        embedded_observations_mask: torch.Tensor,
        embedded_actions: torch.Tensor | None,
    ) -> torch.Tensor:
        """Predict the next action."""
        return TheirPolicy.forward(
            self,
            obs_token=embedded_observations,
            obs_mask=embedded_observations_mask,
            prompt_token=encoded_prompt,
            prompt_token_mask=encoded_prompt_mask,
            action_token=embedded_actions,
        )

    def decode_action_token(
        self, predicted_action_tokens: torch.Tensor
    ) -> dict[PoseActionType, MultiCategorical]:
        """Decode the action token."""
        return TheirPolicy.forward_action_decoder(self, predicted_action_tokens)
