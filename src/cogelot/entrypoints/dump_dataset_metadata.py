from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from tqdm import tqdm

from cogelot.common.io import load_pickle
from cogelot.common.settings import Settings
from cogelot.entrypoints.create_raw_dataset_per_task import get_pickled_instance_paths
from cogelot.structures.vima import Task, VIMAInstance

settings = Settings()


def _get_all_instance_paths(parsed_instances_dir: Path) -> list[Path]:
    """Get all the instance paths from the parsed instances directory."""
    all_instance_paths: list[Path] = []
    for task in Task:
        data_root_for_task: Path = parsed_instances_dir.joinpath(task.name)
        # If there are no instances for that task, we can move on
        if not data_root_for_task.exists():
            continue
        instance_paths = get_pickled_instance_paths(data_root_for_task)
        all_instance_paths.extend(instance_paths)
    return all_instance_paths


def _convert_instance_path_to_metadata(instance_path: Path) -> str:
    """Convert the instance path to metadata."""
    instance = VIMAInstance.model_validate(load_pickle(instance_path))
    return instance.to_metadata().model_dump_json()


def dump_dataset_metadata(
    parsed_instances_dir: Annotated[
        Path, typer.Argument(help="Where to save all of the parsed instances")
    ] = settings.parsed_instances_dir,
    metadata_output_file: Annotated[
        Path, typer.Argument(help="Where to save the metadata")
    ] = settings.dataset_metadata_file,
    *,
    num_workers: Annotated[int, typer.Option(help="Number of workers")] = 1,
) -> None:
    """Dump all the dataset metadata."""
    all_instance_paths = _get_all_instance_paths(parsed_instances_dir)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        logger.info("Submit instance paths for processing")
        futures = {
            executor.submit(_convert_instance_path_to_metadata, path)
            for path in all_instance_paths
        }

        logger.info("Parse and save all the instance metadatta")
        progress_bar = tqdm(desc="Parsing and saving metadata", total=len(all_instance_paths))

        with progress_bar, metadata_output_file.open("w") as metadata_file:
            for future in as_completed(futures):
                metadata_as_json_string: str = future.result()
                metadata_file.write(metadata_as_json_string)
                metadata_file.write("\n")
                progress_bar.update(1)


if __name__ == "__main__":
    dump_dataset_metadata()
