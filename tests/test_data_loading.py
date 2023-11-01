import itertools
from typing import TYPE_CHECKING

from hypothesis import given, strategies as st

from cogelot.data.collate import collate_preprocessed_instances
from cogelot.structures.model import PreprocessedBatch, PreprocessedInstance

if TYPE_CHECKING:
    import torch


def _collate_instances(
    all_preprocessed_instances: list[PreprocessedInstance], batch_size_multiplier: int
) -> PreprocessedBatch:
    all_preprocessed_instances = list(
        itertools.chain.from_iterable(
            [all_preprocessed_instances for _ in range(batch_size_multiplier)]
        )
    )
    batch = collate_preprocessed_instances(all_preprocessed_instances)
    return batch


@given(batch_size_multiplier=st.integers(min_value=1, max_value=10))
def test_collate_preprocessed_instances_does_not_error(
    all_preprocessed_instances: list[PreprocessedInstance], batch_size_multiplier: int
) -> None:
    all_preprocessed_instances = list(
        itertools.chain.from_iterable(
            [all_preprocessed_instances for _ in range(batch_size_multiplier)]
        )
    )
    batch = collate_preprocessed_instances(all_preprocessed_instances)
    assert batch


@given(batch_size_multiplier=st.integers(min_value=1, max_value=10))
def test_number_of_tasks_are_correct(
    all_preprocessed_instances: list[PreprocessedInstance], batch_size_multiplier: int
) -> None:
    desired_batch_size = len(all_preprocessed_instances) * batch_size_multiplier
    batch = _collate_instances(all_preprocessed_instances, batch_size_multiplier)

    assert len(batch.task) == desired_batch_size


@given(batch_size_multiplier=st.integers(min_value=1, max_value=10))
def test_raw_prompt_token_size_has_right_batch_size(
    all_preprocessed_instances: list[PreprocessedInstance], batch_size_multiplier: int
) -> None:
    desired_batch_size = len(all_preprocessed_instances) * batch_size_multiplier
    batch = _collate_instances(all_preprocessed_instances, batch_size_multiplier)

    assert len(batch.raw_prompts_token_type) == desired_batch_size


@given(batch_size_multiplier=st.integers(min_value=1, max_value=10))
def test_prompt_text_has_correct_shape(
    all_preprocessed_instances: list[PreprocessedInstance], batch_size_multiplier: int
) -> None:
    desired_batch_size = len(all_preprocessed_instances) * batch_size_multiplier
    batch = _collate_instances(all_preprocessed_instances, batch_size_multiplier)

    assert batch.word_batch.ndim == 2
    assert batch.word_batch.size(0) == desired_batch_size


@given(batch_size_multiplier=st.integers(min_value=1, max_value=10))
def test_prompt_masks_have_correct_shape(
    all_preprocessed_instances: list[PreprocessedInstance], batch_size_multiplier: int
) -> None:
    desired_batch_size = len(all_preprocessed_instances) * batch_size_multiplier
    batch = _collate_instances(all_preprocessed_instances, batch_size_multiplier)

    max_num_prompt_observations: int = batch.image_batch.to_container()["mask"]["front"].size(1)
    max_num_prompt_objects: int = batch.image_batch.to_container()["mask"]["front"].size(2)

    prompt_masks: dict[str, torch.Tensor] = batch.image_batch.to_container()["mask"]
    for prompt_mask in prompt_masks.values():
        assert prompt_mask.ndim == 3
        assert prompt_mask.size(0) == desired_batch_size
        assert prompt_mask.size(1) == max_num_prompt_observations
        assert prompt_mask.size(2) == max_num_prompt_objects


@given(batch_size_multiplier=st.integers(min_value=1, max_value=10))
def test_prompt_bboxes_have_correct_shape(
    all_preprocessed_instances: list[PreprocessedInstance], batch_size_multiplier: int
) -> None:
    desired_batch_size = len(all_preprocessed_instances) * batch_size_multiplier
    batch = _collate_instances(all_preprocessed_instances, batch_size_multiplier)

    max_num_prompt_observations: int = batch.image_batch.to_container()["mask"]["front"].size(1)
    max_num_prompt_objects: int = batch.image_batch.to_container()["mask"]["front"].size(2)

    prompt_bboxes: dict[str, torch.Tensor] = batch.image_batch.to_container()["bbox"]
    for prompt_bbox in prompt_bboxes.values():
        assert prompt_bbox.ndim == 4
        assert prompt_bbox.size(0) == desired_batch_size
        assert prompt_bbox.size(1) == max_num_prompt_observations
        assert prompt_bbox.size(2) == max_num_prompt_objects
        assert prompt_bbox.size(3) == 4


@given(batch_size_multiplier=st.integers(min_value=1, max_value=10))
def test_prompt_images_have_correct_shape(
    all_preprocessed_instances: list[PreprocessedInstance], batch_size_multiplier: int
) -> None:
    desired_batch_size = len(all_preprocessed_instances) * batch_size_multiplier
    batch = _collate_instances(all_preprocessed_instances, batch_size_multiplier)

    max_num_prompt_observations: int = batch.image_batch.to_container()["mask"]["front"].size(1)
    max_num_prompt_objects: int = batch.image_batch.to_container()["mask"]["front"].size(2)

    prompt_images: dict[str, torch.Tensor] = batch.image_batch.to_container()["cropped_img"]

    for image_per_view in prompt_images.values():
        assert image_per_view.ndim == 6
        assert image_per_view.size(0) == desired_batch_size
        assert image_per_view.size(1) == max_num_prompt_observations
        assert image_per_view.size(2) == max_num_prompt_objects
        assert image_per_view.size(3) == 3
        assert image_per_view.size(4) == 32
        assert image_per_view.size(5) == 32


@given(batch_size_multiplier=st.integers(min_value=1, max_value=10))
def test_actions_have_correct_shape(
    all_preprocessed_instances: list[PreprocessedInstance], batch_size_multiplier: int
) -> None:
    desired_batch_size = len(all_preprocessed_instances) * batch_size_multiplier
    batch = _collate_instances(all_preprocessed_instances, batch_size_multiplier)

    for actions_tensor in batch.actions.to_container().values():
        assert actions_tensor.ndim == 3
        assert actions_tensor.size(0) == desired_batch_size


@given(batch_size_multiplier=st.integers(min_value=1, max_value=10))
def test_observation_end_effector_has_correct_shape(
    all_preprocessed_instances: list[PreprocessedInstance], batch_size_multiplier: int
) -> None:
    desired_batch_size = len(all_preprocessed_instances) * batch_size_multiplier
    batch = _collate_instances(all_preprocessed_instances, batch_size_multiplier)

    max_num_observations: int = batch.observations.to_container()["objects"]["mask"]["top"].size(1)

    end_effector_tensor = batch.observations.to_container()["ee"]
    assert end_effector_tensor.ndim == 3
    assert end_effector_tensor.size(0) == desired_batch_size
    assert end_effector_tensor.size(1) == max_num_observations
    assert end_effector_tensor.size(2) == 1
