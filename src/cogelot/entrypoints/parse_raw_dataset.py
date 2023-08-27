from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, cast

import typer
from loguru import logger
from rich import progress
from tqdm.rich import RateColumn

from cogelot.common.io import save_pickle
from cogelot.data.parse import (
    create_vima_instance_from_instance_dir,
    get_all_raw_instance_directories,
)
from cogelot.entrypoints.settings import Settings
from cogelot.structures.vima import Task


settings = Settings()


def create_progress() -> progress.Progress:
    """Create a progress bar."""
    progress_bar = progress.Progress(
        "[progress.description]{task.description}",
        progress.BarColumn(),
        progress.MofNCompleteColumn(),
        cast(progress.ProgressColumn, RateColumn(unit="it")),
        progress.TimeElapsedColumn(),
        progress.TimeRemainingColumn(),
    )

    return progress_bar


def create_instance_filename_from_raw_instance_dir(instance_dir: Path) -> str:
    """Create the instance filename from the instance dir."""
    task = Task[instance_dir.parent.stem]
    index = int(instance_dir.stem)
    return f"{task.name}/{task.name}_{index}.pkl.gz"


def parse_and_save_instance(
    raw_instance_dir: Path,
    *,
    output_dir: Path,
    replace_if_exists: bool = False,
) -> None:
    """Parse the raw instance and save it to the output dir.

    Because there are repeated values within the instance (as a result of the arrays), each
    instance is also compressed with gzip to make the file size smaller.
    """
    instance_file_name = create_instance_filename_from_raw_instance_dir(raw_instance_dir)
    output_file = output_dir.joinpath(instance_file_name)

    if not replace_if_exists and output_file.exists():
        return

    instance = create_vima_instance_from_instance_dir(raw_instance_dir)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(instance.model_dump(), output_file, compress=True)


def create_vima_instances_from_raw_dataset(
    raw_data_root: Annotated[
        Path, typer.Argument(help="Root directory for the raw data")
    ] = settings.raw_data_dir,
    parsed_data_dir: Annotated[
        Path, typer.Argument(help="Where to save all of the parsed instances")
    ] = settings.parsed_data_dir,
    *,
    num_workers: Annotated[int, typer.Option(help="Number of workers.")] = 2,
    replace_if_exists: Annotated[
        bool, typer.Option(help="Replace parsed instances if they already exist")
    ] = False,
) -> None:
    """Convert the raw VIMA data into parsed instances.

    This means parsing each instance and saving them as a pickle. The reason for doing this is
    because trying to both parse and convert to a HF dataset was crashing way too often and there
    was no way to recover it mid-way through. For a process that can take 8-something hours, we do
    not want to keep re-running this.
    """
    progress_bar = create_progress()
    get_raw_instances_task = progress_bar.add_task("Get all raw instance paths", total=None)
    submit_instance_task = progress_bar.add_task("Submit instances for processing", total=None)
    parsing_instance_task = progress_bar.add_task("Parsing and saving instances", total=None)

    parsed_data_dir.mkdir(parents=True, exist_ok=True)

    with progress_bar:
        logger.info("Get all the raw instance directories")
        all_raw_instance_directories = list(
            progress_bar.track(
                get_all_raw_instance_directories(raw_data_root), task_id=get_raw_instances_task
            )
        )

        # Update the progress bar with the totals
        progress_bar.update(get_raw_instances_task, total=len(all_raw_instance_directories))
        progress_bar.update(submit_instance_task, total=len(all_raw_instance_directories))
        progress_bar.update(parsing_instance_task, total=len(all_raw_instance_directories))

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            logger.info("Submit instance paths for processing")
            futures = {
                executor.submit(
                    parse_and_save_instance,
                    path,
                    output_dir=parsed_data_dir,
                    replace_if_exists=replace_if_exists,
                )
                for path in progress_bar.track(
                    all_raw_instance_directories, task_id=submit_instance_task
                )
            }

            logger.info("Parse and save all the instances")
            for future in as_completed(futures):
                future.result()
                progress_bar.advance(parsing_instance_task)


if __name__ == "__main__":
    create_vima_instances_from_raw_dataset()
