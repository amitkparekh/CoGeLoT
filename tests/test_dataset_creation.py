from pathlib import Path
from typing import cast

import datasets
import torch
from torch.testing._comparison import (
    BooleanPair,
    NonePair,
    NumberPair,
    ObjectPair,
    TensorLikePair,
    not_close_error_metas,
)

from cogelot.data.parse import VIMAInstanceParser
from cogelot.modules.instance_preprocessor import InstancePreprocessor
from cogelot.structures.model import PreprocessedInstance
from cogelot.structures.vima import VIMAInstance


def test_parsing_vima_instance_does_not_error(
    vima_instance_parser: VIMAInstanceParser, raw_instance_dir: Path
) -> None:
    instance = vima_instance_parser(raw_instance_dir)
    assert isinstance(instance, VIMAInstance)
    assert instance.num_observations == instance.num_actions


def test_vima_instance_can_be_saved_into_hf(vima_instance: VIMAInstance) -> None:
    hf_dataset = datasets.Dataset.from_list(
        [vima_instance.model_dump()], features=VIMAInstance.dataset_features()
    )
    hf_dataset = hf_dataset.with_format("torch")
    assert hf_dataset

    # The below has been copy-pasted from torch.testing.assert_close. Unfortunately, this was
    # needed to be able to compare the VIMAInstances since they have tensors and also Python
    # objects. Unfortunately, the thing from torch.testing.assert_close does not have `ObjectPair`
    # as a pair type, so I've put it here manually.
    error_metas = not_close_error_metas(
        actual=VIMAInstance.model_validate(hf_dataset[0]).model_dump(),
        expected=vima_instance.model_dump(),
        pair_types=(
            NonePair,
            BooleanPair,
            NumberPair,
            TensorLikePair,
            ObjectPair,
        ),
        allow_subclasses=True,
        check_dtype=False,
    )
    if error_metas:
        raise error_metas[0].to_error(None)


def test_preprocessing_vima_instances_does_not_error(
    vima_instance: VIMAInstance, instance_preprocessor: InstancePreprocessor
) -> None:
    preprocessed_instance = instance_preprocessor.preprocess(vima_instance)
    assert preprocessed_instance is not None


def test_preprocessed_instance_can_be_saved_into_hf(
    preprocessed_instance: PreprocessedInstance, torch_device: torch.device
) -> None:
    hf_dataset = datasets.Dataset.from_list(
        [preprocessed_instance.model_dump()], features=PreprocessedInstance.dataset_features()
    )
    hf_dataset = hf_dataset.with_format(
        "torch", columns=PreprocessedInstance.hf_tensor_fields, output_all_columns=True
    )
    assert hf_dataset

    # The below has been copy-pasted from torch.testing.assert_close. Unfortunately, this was
    # needed to be able to compare the VIMAInstances since they have tensors and also Python
    # objects. Unfortunately, the thing from torch.testing.assert_close does not have `ObjectPair`
    # as a pair type, so I've put it here manually.
    error_metas = not_close_error_metas(
        actual=PreprocessedInstance.model_validate(hf_dataset[0])
        .transfer_to_device(torch_device)
        .model_dump(),
        expected=preprocessed_instance.model_dump(),
        pair_types=(
            NonePair,
            BooleanPair,
            NumberPair,
            TensorLikePair,
            ObjectPair,
        ),
        allow_subclasses=True,
        check_dtype=False,
    )
    if error_metas:
        raise error_metas[0].to_error(None)


def test_using_seed_when_splitting_is_deterministic(
    vima_instances_dataset: datasets.Dataset,
) -> None:
    seed = 100
    dataset_split_1 = vima_instances_dataset.train_test_split(test_size=0.2, seed=seed)
    dataset_split_2 = vima_instances_dataset.train_test_split(test_size=0.2, seed=seed)

    assert cast(torch.Tensor, dataset_split_1["test"]["task"]).tolist() == [4, 11, 16]
    assert cast(torch.Tensor, dataset_split_2["test"]["task"]).tolist() == [4, 11, 16]


def test_instance_can_be_converted_to_just_metadata(vima_instance: VIMAInstance) -> None:
    instance_metadata = vima_instance.to_metadata()
    assert instance_metadata
