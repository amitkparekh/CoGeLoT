import itertools
from typing import Any, Literal, TypedDict, get_args

import torch
from torch.nn.utils.rnn import pad_sequence

from cogelot.structures.model import PreprocessedBatch, PreprocessedInstance
from cogelot.structures.vima import PoseActionType, Task
from vima.utils import any_to_datadict


ImageFeatureName = Literal["bbox", "cropped_img", "mask"]
ViewLiteral = Literal["front", "top"]
ImageFeatures = dict[ImageFeatureName, dict[ViewLiteral, torch.Tensor]]


class Observation(TypedDict):
    """Structure for the observation."""

    ee: torch.Tensor
    objects: ImageFeatures


def collate_variable_ndim_batch(
    batch: list[torch.Tensor], padding_value: float = 0
) -> torch.Tensor:
    """Collate tensors with multiple dimensions of variable lengths into a single tensor.

    All the tensors need to have the same number of dims, otherwise it will throw an error.
    """
    # [instances, dims]
    shape_per_tensor = torch.tensor([i.shape for i in batch], device=batch[0].device)

    # Get the max size per dimension across all tensors
    max_size_per_dim: list[int] = shape_per_tensor.max(dim=0).values  # noqa: PD011

    # [instances, dims]
    padding_needed_per_tensor: torch.Tensor = max_size_per_dim - shape_per_tensor

    # If there is no padding needed whatsoever, then we can just stack the tensors
    if torch.all(padding_needed_per_tensor == 0):
        return torch.stack(batch, dim=0)

    padded_tensors: list[torch.Tensor] = []
    for tensor, padding_needed_per_dim in zip(batch, padding_needed_per_tensor, strict=True):
        # If there is no padding needed, just append the tensor and continue
        if torch.all(padding_needed_per_dim == 0):
            padded_tensors.append(tensor)
            continue

        # Since we are just going to be padding in one dimension, we want it to be in the form of
        # (0, i, 0, j, ..., 0, k), where i, j, k are the num of padding needed per dim. We can do
        # this using tensor operations to prevent another for-loop here, which is cool.
        padding: list[int] = (
            torch.stack([padding_needed_per_dim, torch.zeros_like(padding_needed_per_dim)])
            .rot90()
            .fliplr()
            .flatten()
            .long()
        ).tolist()

        # Pad the tensor
        padded_tensor = torch.nn.functional.pad(tensor, padding, value=padding_value)
        padded_tensors.append(padded_tensor)

    # Stack the padded tensors
    stacked_tensors = torch.stack(padded_tensors, dim=0)
    return stacked_tensors


def collate_image_features(image_batches: list[ImageFeatures]) -> ImageFeatures:
    """Collate features from multiple images into a single image feature."""
    output: ImageFeatures = {}

    for image_feature_name in get_args(ImageFeatureName):
        output[image_feature_name] = {}
        for view_name in get_args(ViewLiteral):
            output[image_feature_name][view_name] = collate_variable_ndim_batch(
                [image_batch[image_feature_name][view_name] for image_batch in image_batches]
            )
    return output


def collate_action_batch(
    batches: list[dict[PoseActionType, torch.Tensor]], padding_value: int = -100
) -> dict[PoseActionType, torch.Tensor]:
    """Collate the actions across an entire batch."""
    output: dict[PoseActionType, torch.Tensor] = {}

    for pose_action_type in get_args(PoseActionType):
        output[pose_action_type] = collate_variable_ndim_batch(
            [batch[pose_action_type] for batch in batches], padding_value=padding_value
        )
    return output


def collate_observation_batch(observations: list[Observation]) -> Observation:
    """Collate the observations across instances."""
    end_effectors = collate_variable_ndim_batch(
        [observation["ee"] for observation in observations]
    )
    objects = collate_image_features([observation["objects"] for observation in observations])
    return Observation(ee=end_effectors, objects=objects)


def collate_preprocessed_instances(instances: list[PreprocessedInstance]) -> PreprocessedBatch:
    """Collate preprocessed instances into a batch."""
    tasks: list[Task] = [instance.task for instance in instances]
    raw_prompt_token_type = list(
        itertools.chain.from_iterable(instance.raw_prompts_token_type for instance in instances)
    )
    word_batch = pad_sequence([instance.word_batch for instance in instances], batch_first=True)
    image_batch = collate_image_features(
        [instance.image_batch.to_container() for instance in instances]
    )
    actions = collate_action_batch([instance.actions.to_container() for instance in instances])
    observations = collate_observation_batch(
        [Observation(**instance.observations.to_container()) for instance in instances]
    )

    return PreprocessedBatch(
        task=tasks,
        raw_prompts_token_type=raw_prompt_token_type,
        word_batch=word_batch,
        image_batch=any_to_datadict(image_batch),
        actions=any_to_datadict(actions),
        observations=any_to_datadict(observations),
    )


def collate_preprocessed_instances_from_hf_dataset(
    instances: list[dict[str, Any]]
) -> PreprocessedBatch:
    """Collate a batch of preprocessed instances from the HF dataset."""
    parsed_instances = list(map(PreprocessedInstance.from_hf_dict, instances))
    return collate_preprocessed_instances(parsed_instances)
