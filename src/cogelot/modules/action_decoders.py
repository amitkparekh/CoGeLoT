import abc
from typing import cast

import torch
from einops import rearrange

from cogelot.data.collate import collate_variable_ndim_batch
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
    def forward(self, transformer_output: torch.Tensor, *, max_num_objects: int) -> torch.Tensor:
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

    def forward(self, transformer_output: torch.Tensor, *, max_num_objects: int) -> torch.Tensor:
        """Process the output as in the VIMA paper."""
        predicted_action_tokens = transformer_output[:, max_num_objects - 1 :: max_num_objects + 1]
        predicted_action_distributions = self._action_decoder(predicted_action_tokens)

        # Shape (axes, batch size, num action tokens per timestep, dim)
        logits = self._convert_distributions_to_logits(predicted_action_distributions)
        return logits

    def _convert_distributions_to_logits(
        self, predicted_action_distributions: dict[PoseActionType, MultiCategorical]
    ) -> torch.Tensor:
        """Convert the action distributions to a single logits tensor.

        For efficiency, we are going to turn all of the various action distributions into a single
        tensor instead of doing it over and over later on. And no loops because otherwise thats
        slow.
        """
        logits_list = [
            axis.logits
            for _, action_dist in sorted(predicted_action_distributions.items())
            for axis in action_dist.dists
        ]
        logits_list = cast(list[torch.Tensor], logits_list)

        # To prevent the model from incorrectly attending to the padded logits, we set those values
        # to be absolutely tiny but still > 0, so not to contribute to the overall loss
        logits = collate_variable_ndim_batch(
            logits_list, padding_value=torch.finfo(logits_list[0].dtype).tiny
        )
        # Shape (axes, batch size, num action tokens per timestep, dim)
        return logits


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

    def forward(self, transformer_output: torch.Tensor, *, max_num_objects: int) -> torch.Tensor:
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
        logits = rearrange(
            logits_per_token_per_timestep,
            "bsz timesteps action_tokens action_bins -> action_tokens bsz timesteps action_bins",
        )
        return logits
