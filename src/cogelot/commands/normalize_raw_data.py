from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torchdata.dataloader2 import DataLoader2 as DataLoader, MultiProcessingReadingService
from torchdata.datapipes.iter import IterableWrapper
from tqdm.rich import RateColumn

from cogelot.data.vima import VIMAInstance, VIMAInstanceFactory


if TYPE_CHECKING:
    from collections.abc import Iterator


progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    TaskProgressColumn(),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    RateColumn(unit="it"),  # pyright: ignore[reportGeneralTypeIssues]
)


def get_all_instance_directories(raw_data_dir: Path) -> Iterator[Path]:
    """Get all the instance directories."""
    return (
        instance_dir
        for task_dir in raw_data_dir.iterdir()
        for instance_dir in task_dir.iterdir()
        if instance_dir.is_dir()
    )


def file_path_from_instance_dir(instance: VIMAInstance, normalized_path_dir: Path) -> Path:
    """Create the output path from the VIMA instance."""
    return normalized_path_dir.joinpath(f"{instance.task}_{instance.index}.json")


def normalize_instance(instance_dir: Path, normalized_data_dir: Path) -> None:
    """Normalize a single instance and save it."""
    task_number = instance_dir.stem
    task_name = instance_dir.parent.stem
    output_file = normalized_data_dir.joinpath(f"{task_name}_{task_number}.json")

    # Skip instances that have already been normalized
    if output_file.exists():
        return

    instance_factory = VIMAInstanceFactory()
    instance = instance_factory.parse_from_instance_dir(instance_dir)
    output_file.write_text(instance.json())


def new_normalize_raw_data(
    raw_data_dir: Path, normalized_data_dir: Path, *, num_workers: int = 4
) -> None:
    """Normalize all the raw training data into a simple, memory-efficient, and consistent form."""
    save_to_file_fn = partial(file_path_from_instance_dir, normalized_path_dir=normalized_data_dir)

    logger.info("Normalizing instances")
    normalize_progress_task = progress.add_task("Normalizing instances", total=None)

    all_instance_dirs = list(get_all_instance_directories(raw_data_dir))
    datapipe = (
        IterableWrapper(all_instance_dirs)
        .sharding_filter()
        .map(VIMAInstanceFactory().parse_from_instance_dir)
        .map(lambda instance: instance.json())
        .save_to_file(filepath_fn=save_to_file_fn)
    )

    data_loader = DataLoader(
        datapipe, reading_service=MultiProcessingReadingService(num_workers=num_workers)
    )

    with progress:
        for _ in data_loader:
            progress.advance(normalize_progress_task)

    logger.info("Finished normalizing all instances")
    data_loader.shutdown()


def normalize_raw_data(raw_data_dir: Path, normlized_data_dir: Path) -> None:
    """Normalize all the raw training data into a simple, memory-efficient, and consistent form."""
    logger.debug("Getting all the possible instance directories")
    all_instance_dirs = [
        instance_dir for task_dir in raw_data_dir.iterdir() for instance_dir in task_dir.iterdir()
    ]
    logger.info("Normalizing instances")
    normalize_progress_task = progress.add_task(
        "Normalizing instances", total=len(all_instance_dirs)
    )

    failed_instances = []

    with progress:
        for instance_dir in all_instance_dirs:
            try:
                normalize_instance(instance_dir, normlized_data_dir)
            except Exception:  # noqa: BLE001
                failed_instances.append(instance_dir)
                logger.exception(f"Failed to normalize `{instance_dir}`")

            progress.advance(normalize_progress_task)
        logger.debug("Finished normalizing instances")

    logger.info("Finished normalizing all instances")


if __name__ == "__main__":
    raw_data_dir = Path("storage/data/raw/vima_v6/")

    # Create the directory for normalized data if it doesnt exist
    normlized_data_dir = Path("storage/data/normalized/vima_v6/")
    normlized_data_dir.mkdir(parents=True, exist_ok=True)

    # Normalize it
    normalize_raw_data(raw_data_dir, normlized_data_dir)
