import abc

import torch
from einops.einops import rearrange

from cogelot.structures.vima import PoseActionType
from vima import nn as vnn
from vima.nn.action_decoder.dists import MultiCategorical


class ActionDecoder(abc.ABC, torch.nn.Module):
    """Decoder actions from the transformer output into their final form."""

    @property
    @abc.abstractmethod
    def num_action_tokens_per_timestep(self) -> int:
        """The supported number of action tokens per timestep."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(
        self, transformer_output: torch.Tensor, *, max_num_objects: int
    ) -> dict[PoseActionType, MultiCategorical]:
        """Process the transformer output into a distribution for each pose action type."""
        raise NotImplementedError


class VIMAActionDecoder(ActionDecoder):
    """Decode actions the way that was done in the VIMA paper."""

    def __init__(self, action_decoder: vnn.ActionDecoder) -> None:
        super().__init__()
        self._action_decoder = action_decoder

    @property
    def num_action_tokens_per_timestep(self) -> int:
        """The supported number of action tokens per timestep."""
        return 1

    def forward(
        self, transformer_output: torch.Tensor, *, max_num_objects: int
    ) -> dict[PoseActionType, MultiCategorical]:
        """Process the output as in the VIMA paper."""
        predicted_action_tokens = transformer_output[:, max_num_objects - 1 :: max_num_objects + 1]
        predicted_action_distributions = self._action_decoder(predicted_action_tokens)
        return predicted_action_distributions


class TokenPerAxisActionDecoder(ActionDecoder):
    """Decode actions by predicting a token per axis."""

    def __init__(
        self, *, input_dim: int, max_num_action_bins: int, num_action_tokens_per_timestep: int = 14
    ) -> None:
        super().__init__()
        self._max_num_action_bins = max_num_action_bins
        self._num_action_tokens_per_timestep = num_action_tokens_per_timestep

        self.projection = torch.nn.Linear(input_dim, max_num_action_bins)

    @property
    def num_action_tokens_per_timestep(self) -> int:
        """The supported number of action tokens per timestep."""
        return self._num_action_tokens_per_timestep

    def forward(
        self, transformer_output: torch.Tensor, *, max_num_objects: int
    ) -> dict[PoseActionType, MultiCategorical]:
        """Process the output from an autoregressive decoder."""
        num_tokens_per_timestep = max_num_objects + self.num_action_tokens_per_timestep

        transformer_output_per_timestep = rearrange(
            transformer_output,
            "bsz (timesteps num_tokens_per_timestep) dim -> bsz timesteps num_tokens_per_timestep dim",
            num_tokens_per_timestep=num_tokens_per_timestep,
        )

        # Shape (batch size, timesteps, num action tokens per timestep, embed dim)
        action_tokens_from_transformer = transformer_output_per_timestep[
            :, :, -self.num_action_tokens_per_timestep :
        ]

        # Shape (batch size, timesteps, num action tokens per timestep, max num action bins)
        logits_per_token_per_timestep: torch.Tensor = self.projection(
            action_tokens_from_transformer
        )

        predicted_action_distributions = self._convert_to_multicategorical_dists(
            logits_per_token_per_timestep
        )
        return predicted_action_distributions

    def _convert_to_multicategorical_dists(
        self, logits_per_token_per_timestep: torch.Tensor
    ) -> dict[PoseActionType, MultiCategorical]:
        action_dim = logits_per_token_per_timestep.size(-1)
        position_action_dims = [action_dim, action_dim, action_dim]
        rotation_action_dims = [action_dim, action_dim, action_dim, action_dim]
        rearrange_pattern = "B T N D -> B T (N D)"
        pose0_position_logits = rearrange(
            logits_per_token_per_timestep[:, :, :3], rearrange_pattern
        )
        pose0_rotation_logits = rearrange(
            logits_per_token_per_timestep[:, :, 3:7], rearrange_pattern
        )
        pose1_position_logits = rearrange(
            logits_per_token_per_timestep[:, :, 7:10], rearrange_pattern
        )
        pose1_rotation_logits = rearrange(
            logits_per_token_per_timestep[:, :, 10:], rearrange_pattern
        )

        return {
            "pose0_position": MultiCategorical(
                logits=pose0_position_logits, action_dims=position_action_dims
            ),
            "pose0_rotation": MultiCategorical(
                logits=pose0_rotation_logits, action_dims=rotation_action_dims
            ),
            "pose1_position": MultiCategorical(
                logits=pose1_position_logits, action_dims=position_action_dims
            ),
            "pose1_rotation": MultiCategorical(
                logits=pose1_rotation_logits, action_dims=rotation_action_dims
            ),
        }
