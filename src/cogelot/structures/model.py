from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple


if TYPE_CHECKING:
    import torch

    from cogelot.structures.token import ImageToken, ObservationToken, PoseActionToken, TextToken


class PreprocessedInstance(NamedTuple):
    """Preprocessed instance for the model.

    Given a prompt and a history, the model should be able to produce the target. Since
    tokenization is only ever needed once, we just do this aspect once.
    """

    prompt: list[TextToken | ImageToken]
    history: list[ObservationToken | PoseActionToken]
    target: PoseActionToken


class ModelInstance(NamedTuple):
    """Instance directly given to the model."""

    encoder_input: torch.Tensor
    encoder_input_mask: torch.Tensor

    decoder_input: torch.Tensor
    decoder_input_mask: torch.Tensor
