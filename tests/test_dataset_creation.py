from pathlib import Path

import datasets
from torch.testing._comparison import (
    BooleanPair,
    NonePair,
    NumberPair,
    ObjectPair,
    TensorLikePair,
    not_close_error_metas,
)

from cogelot.data.parse import create_vima_instance_from_instance_dir
from cogelot.modules.instance_preprocessor import InstancePreprocessor
from cogelot.structures.model import PreprocessedInstance
from cogelot.structures.vima import VIMAInstance


def test_parsing_vima_instance_does_not_error(raw_instance_dir: Path) -> None:
    instance = create_vima_instance_from_instance_dir(raw_instance_dir)
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
    preprocessed_instance: PreprocessedInstance,
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
        actual=PreprocessedInstance.model_validate(hf_dataset[0]).model_dump(),
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
