from pathlib import Path
from typing import Any, Iterator

from datasets import Dataset
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper

from cogelot.common.io import load_pickle, save_pickle
from cogelot.data.datasets import create_hf_dataset, create_validation_split, dataloader_collate_fn
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
    ds = ds.with_format("torch")

    assert ds
    assert ds[0]


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


def test_dataset_works_with_dataloader(hf_dataset: Dataset) -> None:
    assert hf_dataset

    dataloader = DataLoader(
        hf_dataset,  # pyright: ignore[reportGeneralTypeIssues]
        batch_size=2,
        collate_fn=dataloader_collate_fn,
    )

    # Each batch should be a list of PreprocessedInstance's
    for batch in dataloader:
        assert isinstance(batch, list)

        for instance in batch:
            assert isinstance(instance, PreprocessedInstance)
        break
