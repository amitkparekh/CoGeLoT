import itertools
from pathlib import Path
from typing import Any, Iterator

import datasets
from pytest_cases import fixture

from cogelot.data.datasets import create_hf_dataset, set_dataset_format
from cogelot.data.parse import (
    create_vima_instance_from_instance_dir,
    get_all_raw_instance_directories,
)
from cogelot.data.preprocess import InstancePreprocessor
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

    def gen(preprocessed_instances) -> Iterator[dict[str, Any]]:
        generator = (instance.to_hf_dict() for instance in preprocessed_instances)
        yield from generator

    dataset = create_hf_dataset(gen, all_preprocessed_instances)
    dataset = set_dataset_format(dataset)
    return dataset


# @fixture(scope="session")
# def vima_datamodule(
#     instance_preprocessor: InstancePreprocessor, fixture_storage_dir: Path, tmp_path: Path
# ) -> VIMADataModule:
#     """VIMA datamodule."""
#     raw_data_dir = fixture_storage_dir
#     normalized_data_dir = tmp_path.joinpath("normalized")
#     preprocessed_data_dir = tmp_path.joinpath("preprocessed")
#     datamodule = VIMADataModule(
#         instance_preprocessor=instance_preprocessor,
#         raw_data_dir=raw_data_dir,
#         normalized_data_dir=normalized_data_dir,
#         preprocessed_data_dir=preprocessed_data_dir,
#         num_workers=1,
#         batch_size=1,
#         num_validation_instances=0,
#     )
#     datamodule.prepare_data()
#     # TODO: Increase the total number of instances

#     datamodule.setup(stage="fit")

#     assert datamodule.training_datapipe is not None
#     assert datamodule.validation_datapipe is not None

#     return datamodule
