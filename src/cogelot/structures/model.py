from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple


if TYPE_CHECKING:
    import torch

    from vima.utils import DataDict


class PreprocessedInstance(NamedTuple):
    """Preprocessed instance for the model."""

    prompt: tuple[list[list[int]], torch.Tensor, DataDict]
    observations: DataDict
    actions: DataDict


class ModelInstance(NamedTuple):
    """Instance directly given to the model."""

    encoder_input: torch.Tensor
    encoder_input_mask: torch.Tensor

    decoder_input: torch.Tensor
    decoder_input_mask: torch.Tensor
