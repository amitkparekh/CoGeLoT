from typing import NamedTuple

import torch

from vima.utils import DataDict


class PreprocessedInstance(NamedTuple):
    """Preprocessed instance for the model.

    Given a prompt and a history, the model should be able to produce the target. Since
    tokenization is only ever needed once, we just do this aspect once.
    """

    prompt: tuple[list[list[int]], torch.Tensor, DataDict]
    observations: DataDict
    actions: DataDict


class ModelInstance(NamedTuple):
    """Instance directly given to the model."""

    embedded_prompt: torch.Tensor
    embedded_prompt_mask: torch.Tensor

    embedded_observations: torch.Tensor
    embedded_observations_mask: torch.Tensor

    embedded_actions: torch.Tensor
