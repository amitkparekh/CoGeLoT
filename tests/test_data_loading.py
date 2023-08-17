import datasets

from cogelot.data.collate import collate_preprocessed_instances
from cogelot.datamodules import VIMATrainingDataModule
from cogelot.structures.model import PreprocessedInstance


def test_collate_preprocessed_instances_works(hf_dataset: datasets.Dataset) -> None:
    preprocessed_instances = list(map(PreprocessedInstance.from_hf_dict, hf_dataset))  # type: ignore

    batch = collate_preprocessed_instances(preprocessed_instances)
    assert batch


def test_vima_training_datamodule_loads_without_error(
    vima_training_datamodule: VIMATrainingDataModule,
) -> None:
    assert vima_training_datamodule
