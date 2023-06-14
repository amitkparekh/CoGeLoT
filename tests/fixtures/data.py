from pathlib import Path

from pytest_cases import fixture

from cogelot.data.normalize import create_vima_instance_from_instance_dir
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
