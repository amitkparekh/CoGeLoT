from pathlib import Path
from typing import Any, Iterator

from datasets import Dataset
from torchdata.datapipes.iter import IterableWrapper

from cogelot.common.io import load_pickle, save_pickle
from cogelot.data.hf_dataset import create_hf_dataset, create_validation_split
from cogelot.data.parse import create_vima_instance_from_instance_dir
from cogelot.data.preprocess import InstancePreprocessor
from cogelot.structures.model import PreprocessedInstance
from cogelot.structures.vima import VIMAInstance


def test_raw_data_parsing_works(raw_instance_dir: Path) -> None:
    instance = create_vima_instance_from_instance_dir(raw_instance_dir)
    assert isinstance(instance, VIMAInstance)
    assert instance.num_observations == instance.num_actions


def test_saving_vima_instance_works(vima_instance: VIMAInstance, tmp_path: Path) -> None:
    output_path = vima_instance.save(tmp_path)
    assert output_path.is_file()
    assert VIMAInstance.load(output_path)

    output_path = vima_instance.save(tmp_path, compress=True)
    assert output_path.is_file()
    assert VIMAInstance.load(output_path)


def test_create_hf_dataset_from_vima_instance_works(fixture_storage_dir: Path) -> None:
    def gen_from_task_dir() -> Iterator[dict[str, Any]]:
        for instance_dir in fixture_storage_dir.glob("*/*/"):
            instance = create_vima_instance_from_instance_dir(instance_dir)
            yield instance.dict()

    dataset = Dataset.from_generator(gen_from_task_dir)
    assert dataset


def test_preprocessing_data_works(
    normalized_instance: VIMAInstance, instance_preprocessor: InstancePreprocessor
) -> None:
    preprocessed_instance = instance_preprocessor.preprocess(normalized_instance)
    assert preprocessed_instance is not None


def test_saving_preprocessed_instance_works(
    preprocessed_instance: PreprocessedInstance, tmp_path: Path
) -> None:
    saved_path = save_pickle(preprocessed_instance, tmp_path.joinpath("1.pkl"))
    assert saved_path.is_file()
    assert load_pickle(saved_path)


def test_create_hf_dataset(all_preprocessed_instances: list[PreprocessedInstance]) -> None:
    def gen() -> Iterator[dict[str, Any]]:
        for instance in all_preprocessed_instances:
            yield instance.to_hf_dict()

    ds = create_hf_dataset(gen)

    assert ds


def test_validation_split_creation_works(
    all_preprocessed_instances: list[PreprocessedInstance],
) -> None:
    num_cycles = 5
    num_valid_instances = 2

    # Create a datapipe and repeat the input data multiple times
    all_preprocessed_instances = IterableWrapper(all_preprocessed_instances).cycle(num_cycles)

    def gen() -> Iterator[dict[str, Any]]:
        for instance in all_preprocessed_instances:
            yield instance.to_hf_dict()

    dataset = create_hf_dataset(gen)
    split_dataset = create_validation_split(
        dataset, max_num_validation_instances=num_valid_instances
    )

    assert split_dataset
    assert (
        len(list(split_dataset["train"]))
        == (num_cycles * len(all_preprocessed_instances)) - num_valid_instances
    )
    assert len(list(split_dataset["test"])) == num_valid_instances
