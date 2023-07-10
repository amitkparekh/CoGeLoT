from typing import Any, NamedTuple

import torch
from pydantic import BaseModel

from cogelot.structures.vima import Task
from vima.utils import DataDict


class PreprocessedInstance(BaseModel, arbitrary_types_allowed=True):
    """Preprocessed instance for the model.

    Given a prompt and a history, the model should be able to produce the target. Since
    tokenization is only ever needed once, we just do this aspect once.
    """

    task: Task

    raw_prompts_token_type: list[list[int]]
    word_batch: torch.Tensor
    image_batch: DataDict

    observations: DataDict
    actions: DataDict

    def to_hf_dict(self) -> dict[str, Any]:
        """To a dictionary for HF datasets."""
        return {
            "task": self.task,
            "raw_prompts_token_type": self.raw_prompts_token_type,
            "word_batch": self.word_batch,
            "image_batch": self.image_batch.to_container(),
            "observations": self.observations.to_container(),
            "actions": self.actions.to_container(),
        }


class ModelInstance(NamedTuple):
    """Instance directly given to the model."""

    embedded_prompt: torch.Tensor
    embedded_prompt_mask: torch.Tensor

    embedded_observations: torch.Tensor
    embedded_observations_mask: torch.Tensor

    embedded_actions: torch.Tensor
