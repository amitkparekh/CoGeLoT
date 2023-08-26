from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

import typer
from rich.progress import track

from cogelot.common.io import save_pickle
from cogelot.data.parse import (
    create_vima_instance_from_instance_dir,
    get_all_raw_instance_directories,
)
from cogelot.entrypoints.settings import Settings


settings = Settings()


def get_raw_instance_directories(raw_data_root: Path) -> list[Path]:
    """Get all the raw instance directories."""
    path_iterator = get_all_raw_instance_directories(raw_data_root)
    all_paths = list(track(path_iterator, description="Getting all raw instance paths"))
    return all_paths


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
    instance = create_vima_instance_from_instance_dir(raw_instance_dir)

    instance_file_name = f"{instance.task.name}/{instance.task.name}_{instance.index}.pkl.gz"
    output_file = output_dir.joinpath(instance_file_name)

    if not replace_if_exists and output_file.exists():
        return

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
    all_raw_instance_directories = get_raw_instance_directories(raw_data_root)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                parse_and_save_instance,
                path,
                output_dir=parsed_data_dir,
                replace_if_exists=replace_if_exists,
            )
            for path in all_raw_instance_directories
        }
        tracked_iterator = track(
            as_completed(futures), description="Parsing and saving instances", total=len(futures)
        )
        for future in tracked_iterator:
            future.result()
