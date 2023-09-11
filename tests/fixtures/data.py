import itertools
from pathlib import Path

import datasets
from pytest_cases import fixture

from cogelot.data.parse import (
    create_vima_instance_from_instance_dir,
    get_all_raw_instance_directories,
)
from cogelot.modules.instance_preprocessor import InstancePreprocessor
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
def all_vima_instances(fixture_storage_dir: Path) -> list[VIMAInstance]:
    """All normalized VIMA instances."""
    parsed_instances = (
        create_vima_instance_from_instance_dir(instance_dir)
        for instance_dir in get_all_raw_instance_directories(fixture_storage_dir)
    )
    return list(parsed_instances)


@fixture(scope="session")
def vima_instances_dataset(all_vima_instances: list[VIMAInstance]) -> datasets.Dataset:
    """All normalized VIMA instances."""
    dataset = datasets.Dataset.from_list(
        [instance.model_dump() for instance in all_vima_instances],
        features=VIMAInstance.dataset_features(),
    )
    dataset = dataset.with_format("torch")
    return dataset


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
def preprocessed_instances_dataset(
    all_preprocessed_instances: list[PreprocessedInstance],
) -> datasets.Dataset:
    # Repeat the input data multiple times
    num_cycles = 5
    all_preprocessed_instances = list(
        itertools.chain.from_iterable([all_preprocessed_instances for _ in range(num_cycles)])
    )
    dataset = datasets.Dataset.from_list(
        [instance.model_dump() for instance in all_preprocessed_instances],
        features=PreprocessedInstance.dataset_features(),
    )
    dataset = dataset.with_format(
        "torch", columns=PreprocessedInstance.hf_tensor_fields, output_all_columns=True
    )
    return dataset
