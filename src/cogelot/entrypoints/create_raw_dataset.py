from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Annotated

import datasets
import typer
from loguru import logger

from cogelot.common.io import load_pickle, save_pickle
from cogelot.common.rich import create_progress_bar
from cogelot.data.datasets import (
    create_hf_dataset_from_pickled_instances,
    load_instance_from_pickled_path,
)
from cogelot.data.parse import (
    create_vima_instance_from_instance_dir,
    get_all_raw_instance_directories,
)
from cogelot.entrypoints.settings import Settings
from cogelot.structures.vima import Task, VIMAInstance


settings = Settings()


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


load_vima_instance_from_path_fn = partial(
    load_instance_from_pickled_path,
    instance=VIMAInstance,
    load_from_path_fn=load_pickle,
)


def create_validation_split(
    dataset: datasets.Dataset,
    *,
    max_num_validation_instances: int,
    seed: int = 0,
    stratify_column: str = "task",
) -> datasets.DatasetDict:
    """Create the validation split for the dataset."""
    dataset_split = dataset.train_test_split(
        test_size=max_num_validation_instances,
        stratify_by_column=stratify_column,
        seed=seed,
    )
    dataset_dict = datasets.DatasetDict(
        {"train": dataset_split["train"], "valid": dataset_split["test"]}
    )
    return dataset_dict


def create_vima_instances_from_raw_dataset(
    raw_data_root: Path,
    parsed_instances_dir: Path,
    *,
    num_workers: int,
    replace_if_exists: bool = False,
) -> None:
    """Convert the raw VIMA data into parsed instances.

    This means parsing each instance and saving them as a pickle. The reason for doing this is
    because trying to both parse and convert to a HF dataset was crashing way too often and there
    was no way to recover it mid-way through. For a process that can take 8-something hours, we do
    not want to keep re-running this.
    """
    progress_bar = create_progress_bar()
    get_raw_instances_task = progress_bar.add_task("Get all raw instance paths", total=None)
    submit_instance_task = progress_bar.add_task("Submit instances for processing", total=None)
    parsing_instance_task = progress_bar.add_task("Parsing and saving instances", total=None)

    parsed_instances_dir.mkdir(parents=True, exist_ok=True)

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
                    output_dir=parsed_instances_dir,
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


def convert_vima_instance_to_hf_dataset(
    parsed_instances_dir: Path,
    parsed_hf_dataset_per_task_dir: Path,
    parsed_hf_dataset_dir: Path,
    num_workers: int,
    num_validation_instances: int,
    max_shard_size: str,
    seed: int,
) -> None:
    """Convert the parsed VIMA instances to a dataset on HF.

    We load all the pickled files, parse them to ensure they're correct, turn them into the HF
    dataset, and upload it to HF.
    """
    dataset = create_hf_dataset_from_pickled_instances(
        parsed_instances_dir,
        temp_output_dir=parsed_hf_dataset_per_task_dir,
        num_workers=num_workers,
        max_shard_size=max_shard_size,
        writer_batch_size=settings.writer_batch_size,
        dataset_features=VIMAInstance.dataset_features(),
        load_instance_from_path_fn=load_vima_instance_from_path_fn,
    )

    logger.info("Creating the train-valid split...")
    dataset_with_split = create_validation_split(
        dataset, max_num_validation_instances=num_validation_instances, seed=seed
    )

    logger.info("Saving the dataset to disk...")
    dataset_with_split.save_to_disk(
        parsed_hf_dataset_dir, max_shard_size=max_shard_size, num_proc=None
    )


def create_raw_dataset(
    raw_data_root: Annotated[
        Path, typer.Argument(help="Root directory for the raw data")
    ] = settings.raw_data_dir,
    parsed_data_dir: Annotated[
        Path, typer.Argument(help="Where to save all of the parsed instances")
    ] = settings.parsed_data_dir,
    parsed_hf_dataset_dir: Annotated[
        Path, typer.Argument(help="Where to save the parsed HF dataset")
    ] = settings.parsed_hf_dataset_dir,
    parsed_hf_dataset_per_task_dir: Annotated[
        Path, typer.Argument(help="Where to save the HF dataset for each task")
    ] = settings.parsed_hf_dataset_per_task_dir,
    *,
    num_workers: Annotated[int, typer.Option(help="Number of workers.")] = 1,
    replace_if_exists: Annotated[
        bool, typer.Option(help="Replace parsed instances if they already exist")
    ] = False,
    num_validation_instances: Annotated[
        int,
        typer.Option(
            help="Maximum number of validation instances, created using stratefied sampling"
        ),
    ] = settings.num_validation_instances,
    max_shard_size: Annotated[
        str, typer.Option(help="Maximum shard size for the dataset")
    ] = settings.max_shard_size,
    seed: Annotated[int, typer.Option(help="Seed for the stratified sampling.")] = 1000,
) -> None:
    """Create the raw dataset from the raw VIMA data, and upload to HF.

    This happens in two steps: parsing the raw data into instances, and then converting the
    instances into the HF dataset.

    After parsing the raw data into instances, we save this to a folder because when doing it in
    one step, the HF dataset creation process was crashing and since it takes hours to do, we don't
    want to keep re-doing that.
    """
    create_vima_instances_from_raw_dataset(
        raw_data_root=raw_data_root,
        parsed_instances_dir=parsed_data_dir,
        num_workers=num_workers,
        replace_if_exists=replace_if_exists,
    )
    convert_vima_instance_to_hf_dataset(
        parsed_instances_dir=parsed_data_dir,
        parsed_hf_dataset_per_task_dir=parsed_hf_dataset_per_task_dir,
        parsed_hf_dataset_dir=parsed_hf_dataset_dir,
        num_workers=num_workers,
        num_validation_instances=num_validation_instances,
        max_shard_size=max_shard_size,
        seed=seed,
    )


if __name__ == "__main__":
    create_raw_dataset()
