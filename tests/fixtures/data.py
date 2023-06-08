import pickle
from pathlib import Path
from typing import Any

from pytest_cases import fixture

from cogelot.data.vima import VIMAInstance, VIMAInstanceFactory


@fixture(scope="session")
def fixture_storage_dir() -> Path:
    """Fixture storage directory."""
    return Path("storage/fixtures")


@fixture(scope="session")
def data_dir(fixture_storage_dir: Path, mission_task: str, mission_id: str) -> Path:
    """Fixture storage directory."""
    return fixture_storage_dir.joinpath(mission_task, f"{mission_id}/")


@fixture(scope="session")
def vima_instance(data_dir: Path) -> VIMAInstance:
    """Load example data."""
    vima_instance_factory = VIMAInstanceFactory()
    instance = vima_instance_factory.parse_from_instance_dir(data_dir)
    return instance


@fixture(scope="session")
def trajectory_metadata(data_dir: Path) -> dict[str, Any]:
    """Load example data."""
    trajectory_metadata = pickle.load(data_dir.joinpath("trajectory.pkl").open("rb"))
    return trajectory_metadata
