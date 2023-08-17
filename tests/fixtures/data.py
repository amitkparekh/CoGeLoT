import itertools
from pathlib import Path
from typing import Any, Iterator

import datasets
from pyparsing import Iterable
from pytest_cases import fixture

from cogelot.data.datasets import create_hf_dataset, set_dataset_format
from cogelot.data.parse import (
    create_vima_instance_from_instance_dir,
    get_all_raw_instance_directories,
)
from cogelot.data.preprocess import InstancePreprocessor
from cogelot.datamodules.training import VIMATrainingDataModule
from cogelot.structures.model import PreprocessedInstance
from cogelot.structures.vima import VIMAInstance


@fixture(scope="session")
def fixture_storage_dir() -> Path:
    """Fixture storage directory."""
    return Path("storage/fixtures")


@fixture(scope="session")
def raw_instance_dir(fixture_storage_dir: Path, mission_task: str, mission_id: str) -> Path:
    """Fixture storage directory."""
    return fixture_storage_dir.joinpath(mission_task, f"{mission_id}/")


@fixture(scope="session")
def vima_instance(raw_instance_dir: Path) -> VIMAInstance:
    """A single normalized VIMA instance."""
    return create_vima_instance_from_instance_dir(raw_instance_dir)


@fixture(scope="session")
def preprocessed_instance(
    vima_instance: VIMAInstance, instance_preprocessor: InstancePreprocessor
) -> PreprocessedInstance:
    """A single preprocessed instance."""
    return instance_preprocessor.preprocess(vima_instance)


@fixture(scope="session")
def all_preprocessed_instances(
    fixture_storage_dir: Path, instance_preprocessor: InstancePreprocessor
) -> list[PreprocessedInstance]:
    """All preprocessed instances."""
    parsed_instances = (
        create_vima_instance_from_instance_dir(instance_dir)
        for instance_dir in get_all_raw_instance_directories(fixture_storage_dir)
    )
    preprocessed_instances = (
        instance_preprocessor.preprocess(instance) for instance in parsed_instances
    )
    return list(preprocessed_instances)


@fixture(scope="session")
def hf_dataset(all_preprocessed_instances: list[PreprocessedInstance]) -> datasets.Dataset:
    num_cycles = 5

    # Repeat the input data multiple times
    all_preprocessed_instances = list(
        itertools.chain.from_iterable([all_preprocessed_instances for _ in range(num_cycles)])
    )

    def gen(preprocessed_instances: Iterable[PreprocessedInstance]) -> Iterator[dict[str, Any]]:
        generator = (instance.to_hf_dict() for instance in preprocessed_instances)
        yield from generator

    dataset = create_hf_dataset(gen, all_preprocessed_instances)
    dataset = set_dataset_format(dataset)
    return dataset


@fixture(scope="session")
def vima_training_datamodule() -> VIMATrainingDataModule:
    """VIMA datamodule."""
    datamodule = VIMATrainingDataModule(
        hf_datasets_repo_name="amitkparekh/vima-small",
        num_workers=1,
        batch_size=1,
    )
    return datamodule
