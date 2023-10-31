import abc

import torch

from cogelot.structures.vima import PoseActionType
from vima import nn as vnn
from vima.nn.action_decoder.dists import MultiCategorical


class ActionDecoder(abc.ABC, torch.nn.Module):
    """Decoder actions from the transformer output into their final form."""

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

    def forward(
        self, transformer_output: torch.Tensor, *, max_num_objects: int
    ) -> dict[PoseActionType, MultiCategorical]:
        """Process the output as in the VIMA paper."""
        predicted_action_tokens = transformer_output[:, max_num_objects - 1 :: max_num_objects + 1]
        predicted_action_distributions = self._action_decoder(predicted_action_tokens)
        return predicted_action_distributions
