from pathlib import Path

from pytest_cases import fixture

from cogelot.data import datapipes
from cogelot.data.normalize import create_vima_instance_from_instance_dir
from cogelot.data.preprocess import InstancePreprocessor
from cogelot.data.vima_datamodule import VIMADataModule
from cogelot.structures.model import PreprocessedInstance
from cogelot.structures.vima import VIMAInstance


@fixture(scope="session")
def fixture_storage_dir() -> Path:
    """Fixture storage directory."""
    return Path("storage/fixtures")


@fixture(scope="session")
def data_dir(fixture_storage_dir: Path, mission_task: str, mission_id: str) -> Path:
    """Fixture storage directory."""
    return fixture_storage_dir.joinpath(mission_task, f"{mission_id}/")


@fixture(scope="session")
def normalized_instance(data_dir: Path) -> VIMAInstance:
    """A single normalized VIMA instance."""
    return create_vima_instance_from_instance_dir(data_dir)


@fixture(scope="session")
def preprocessed_instance(
    normalized_instance: VIMAInstance, instance_preprocessor: InstancePreprocessor
) -> PreprocessedInstance:
    """A single preprocessed instance."""
    return instance_preprocessor.preprocess(normalized_instance)


@fixture(scope="module")
def all_preprocessed_instances(
    fixture_storage_dir: Path, instance_preprocessor: InstancePreprocessor
) -> list[PreprocessedInstance]:
    """All preprocessed instances."""
    normalize_datapipe = datapipes.normalize_raw_data(fixture_storage_dir)
    preprocessed_instances = [
        instance_preprocessor.preprocess(instance) for instance in normalize_datapipe
    ]
    return preprocessed_instances


@fixture(scope="session")
def vima_datamodule(
    instance_preprocessor: InstancePreprocessor, fixture_storage_dir: Path, tmp_path: Path
) -> VIMADataModule:
    """VIMA datamodule."""
    raw_data_dir = fixture_storage_dir
    normalized_data_dir = tmp_path.joinpath("normalized")
    preprocessed_data_dir = tmp_path.joinpath("preprocessed")
    datamodule = VIMADataModule(
        instance_preprocessor=instance_preprocessor,
        raw_data_dir=raw_data_dir,
        normalized_data_dir=normalized_data_dir,
        preprocessed_data_dir=preprocessed_data_dir,
        num_workers=1,
        batch_size=1,
        num_validation_instances=0,
    )
    datamodule.prepare_data()
    # TODO: Increase the total number of instances

    datamodule.setup(stage="fit")

    assert datamodule.training_datapipe is not None
    assert datamodule.validation_datapipe is not None

    return datamodule
