from __future__ import annotations

from pathlib import Path

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
from tqdm.rich import RateColumn

from cogelot.data.vima import VIMAInstanceFactory


progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    TaskProgressColumn(),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    RateColumn(unit="it"),  # pyright: ignore[reportGeneralTypeIssues]
)


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
