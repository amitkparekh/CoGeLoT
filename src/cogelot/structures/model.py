from typing import Any, Literal, NamedTuple, Self

import torch
from pydantic import BaseModel

from cogelot.structures.vima import Task
from vima.utils import DataDict, any_to_datadict


RawPromptTokenType = list[list[Literal[0, 1]]]


class PreprocessedInstance(BaseModel, arbitrary_types_allowed=True):
    """Preprocessed instance for the model.

    Given a prompt and a history, the model should be able to produce the target. Since
    tokenization is only ever needed once, we just do this aspect once.
    """

    task: Task

    raw_prompts_token_type: RawPromptTokenType
    word_batch: torch.Tensor
    image_batch: DataDict

    observations: DataDict
    actions: DataDict

    def to_hf_dict(self) -> dict[str, Any]:
        """To a dictionary for HF datasets."""
        return {
            "task": self.task.name,
            "raw_prompts_token_type": self.raw_prompts_token_type,
            "word_batch": self.word_batch,
            "image_batch": self.image_batch.to_container(),
            "observations": self.observations.to_container(),
            "actions": self.actions.to_container(),
        }

    @classmethod
    def from_hf_dict(cls, instance: dict[str, Any]) -> Self:
        """From a dictionary outputted by the HF datasets."""
        return cls(
            task=Task.from_sorted_task_list_index(instance["task"]),
            raw_prompts_token_type=instance["raw_prompts_token_type"],
            word_batch=instance["word_batch"],
            image_batch=any_to_datadict(instance["image_batch"]),
            observations=any_to_datadict(instance["observations"]),
            actions=any_to_datadict(instance["actions"]),
        )

    def transfer_to_device(self, device: torch.device) -> "PreprocessedInstance":
        """Transfer any tensors to the given device."""
        word_batch = self.word_batch.to(device)
        image_batch = self.image_batch.to_torch_tensor(device=device)
        observations = self.observations.to_torch_tensor(device=device)
        actions = self.actions.to_torch_tensor(device=device)

        assert isinstance(image_batch, DataDict)
        assert isinstance(observations, DataDict)
        assert isinstance(actions, DataDict)

        return PreprocessedInstance(
            task=self.task,
            raw_prompts_token_type=self.raw_prompts_token_type,
            word_batch=word_batch,
            image_batch=image_batch,
            observations=observations,
            actions=actions,
        )


class PreprocessedBatch(NamedTuple):
    """Preprocessed Batch that will get made by the collate fn."""

    task: list[Task]

    raw_prompts_token_type: RawPromptTokenType
    word_batch: torch.Tensor
    image_batch: DataDict

    observations: DataDict
    actions: DataDict


class ModelInstance(NamedTuple):
    """Instance directly given to the model."""

    # [batch, tokens, dim]
    encoded_prompt: torch.Tensor
    # [batch, tokens]
    encoded_prompt_mask: torch.Tensor

    # [batch, obs, obj, dim]
    embedded_observations: torch.Tensor
    # [batch, obs, obj]
    embedded_observations_mask: torch.Tensor

    # [batch, actions, dim]
    embedded_actions: torch.Tensor | None
