from typing import Any, ClassVar, Literal, NamedTuple, Self

import datasets
import torch
from pydantic import BaseModel, ConfigDict

from cogelot.structures.common import Bbox, PydanticHFDatasetMixin
from cogelot.structures.token import EndEffectorToken, PoseActionToken, TextToken
from cogelot.structures.vima import Partition, Task
from vima.utils import DataDict, any_to_datadict


RawPromptTokenType = list[list[Literal[0, 1]]]


_RawPromptsTokenType = datasets.Sequence(datasets.Value("int8"))
_CroppedImg = datasets.Array3D(shape=(3, 32, 32), dtype="float32", id="cropped_img")
_Mask = datasets.Value("bool")


def _wrap_feature_in_batch_sequence(feature: Any, *, length: int = 1) -> datasets.Sequence:
    return datasets.Sequence(id="batch", length=length, feature=feature)


def _wrap_feature_in_objects_sequence(feature: Any) -> datasets.Sequence:
    return datasets.Sequence(id="objects", feature=feature)


def _wrap_feature_in_observation_sequence(feature: Any) -> datasets.Sequence:
    return datasets.Sequence(id="obs", feature=feature)


def _image_batch_feature_wrapper(feature: Any) -> datasets.Sequence:
    return datasets.Sequence(id="tokens", feature=_wrap_feature_in_objects_sequence(feature))


def _observation_feature_wrapper(feature: Any) -> datasets.Sequence:
    return _wrap_feature_in_observation_sequence(
        _wrap_feature_in_batch_sequence(_wrap_feature_in_objects_sequence(feature))
    )


class PreprocessedInstance(BaseModel, PydanticHFDatasetMixin):
    """Preprocessed instance for the model.

    Given a prompt and a history, the model should be able to produce the target. Since
    tokenization is only ever needed once, we just do this aspect once.
    """

    hf_tensor_fields: ClassVar[list[str]] = [
        "word_batch",
        "image_batch",
        "observations",
        "actions",
    ]

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    @classmethod
    def dataset_features(cls) -> datasets.Features:
        """Features for the HF dataset."""
        return datasets.Features(
            {
                "task": Task.dataset_feature(),
                "raw_prompts_token_type": _wrap_feature_in_batch_sequence(_RawPromptsTokenType),
                "word_batch": datasets.Sequence(id="tokens", feature=TextToken.dataset_feature()),
                "image_batch": {
                    "bbox": {
                        "front": _image_batch_feature_wrapper(Bbox.dataset_feature()),
                        "top": _image_batch_feature_wrapper(Bbox.dataset_feature()),
                    },
                    "cropped_img": {
                        "front": _image_batch_feature_wrapper(_CroppedImg),
                        "top": _image_batch_feature_wrapper(_CroppedImg),
                    },
                    "mask": {
                        "front": _image_batch_feature_wrapper(_Mask),
                        "top": _image_batch_feature_wrapper(_Mask),
                    },
                },
                "observations": {
                    "ee": datasets.Sequence(datasets.Sequence(EndEffectorToken.dataset_feature())),
                    "objects": {
                        "bbox": {
                            "front": _observation_feature_wrapper(Bbox.dataset_feature()),
                            "top": _observation_feature_wrapper(Bbox.dataset_feature()),
                        },
                        "cropped_img": {
                            "front": _observation_feature_wrapper(_CroppedImg),
                            "top": _observation_feature_wrapper(_CroppedImg),
                        },
                        "mask": {
                            "front": _observation_feature_wrapper(_Mask),
                            "top": _observation_feature_wrapper(_Mask),
                        },
                    },
                },
                "actions": PoseActionToken.dataset_features(),
            }
        )


class EvaluationEpisode(NamedTuple):
    """Single instance of the evaluation dataset."""

    partition: Partition
    task: Task


class PreprocessedBatch(NamedTuple):
    """Preprocessed Batch that will get made by the collate fn."""

    task: list[Task]

    raw_prompts_token_type: RawPromptTokenType
    word_batch: torch.Tensor
    image_batch: DataDict

    observations: DataDict
    actions: DataDict

    def __len__(self) -> int:
        """Length of the batch."""
        return len(self.task)


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
