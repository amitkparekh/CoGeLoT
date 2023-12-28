from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Optional

import typer
from loguru import logger
from tqdm import tqdm

from cogelot.common.io import save_pickle
from cogelot.data.parse import (
    VIMAInstanceParser,
    get_all_raw_instance_directories,
)
from cogelot.entrypoints.settings import Settings
from cogelot.structures.vima import Task, VIMAInstance

settings = Settings()


def _create_vima_instance_parser() -> VIMAInstanceParser:
    """Create the VIMA instance parser, with the correct settings."""
    if settings.dataset_variant == "original":
        return VIMAInstanceParser(keep_null_action=False)
    if settings.dataset_variant == "keep_null_action":
        return VIMAInstanceParser(keep_null_action=True)

    raise ValueError(f"Unknown dataset variant: {settings.dataset_variant}")


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
    create_vima_instance_from_instance_dir: Callable[[Path], VIMAInstance],
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


def parse_original_dataset(
    raw_data_root: Annotated[
        Path, typer.Argument(help="Root directory for the raw data")
    ] = settings.raw_data_dir,
    parsed_instances_dir: Annotated[
        Path, typer.Argument(help="Where to save all of the parsed instances")
    ] = settings.parsed_instances_dir,
    *,
    num_workers: Annotated[int, typer.Option(help="Number of workers")] = 1,
    replace_if_exists: Annotated[
        bool, typer.Option(help="Replace parsed instances if they already exist")
    ] = False,
    task_index_filter: Annotated[
        Optional[int], typer.Option(min=Task.minimum(), max=Task.maximum())  # noqa: UP007
    ] = None,
) -> None:
    """Convert the raw VIMA data into the instances we can work with.

    This means parsing each instance and saving them as a pickle. The reason for doing this is
    because trying to both parse and convert to a HF dataset was crashing way too often and there
    was no way to recover it mid-way through. For a process that can take 8-something hours, we do
    not want to keep re-running this.
    """
    task_filter = Task(task_index_filter) if task_index_filter is not None else None
    logger.info("Get all the raw instance directories")
    all_raw_instance_directories = list(
        tqdm(
            get_all_raw_instance_directories(raw_data_root, task_filter=task_filter),
            desc="Get all raw instance paths",
        )
    )

    vima_instance_parser = _create_vima_instance_parser()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        logger.info("Submit instance paths for processing")
        futures = {
            executor.submit(
                parse_and_save_instance,
                path,
                output_dir=parsed_instances_dir,
                replace_if_exists=replace_if_exists,
                create_vima_instance_from_instance_dir=vima_instance_parser.create_partial(),
            )
            for path in tqdm(all_raw_instance_directories, desc="Submit instances for processing")
        }

        logger.info("Parse and save all the instances")
        progress_bar = tqdm(
            desc="Parsing and saving instances", total=len(all_raw_instance_directories)
        )
        with progress_bar:
            for future in as_completed(futures):
                future.result()
                progress_bar.update()
