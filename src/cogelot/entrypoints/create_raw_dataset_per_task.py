import itertools
from functools import partial
from pathlib import Path
from typing import Annotated

import datasets
import typer
from loguru import logger

from cogelot.common.io import load_pickle
from cogelot.data.datasets import (
    create_hf_dataset_from_paths,
    get_pickled_instance_paths,
    load_instance_from_pickled_path,
)
from cogelot.entrypoints.settings import Settings
from cogelot.structures.vima import Task, VIMAInstance


settings = Settings()


def _get_config_name_for_task(task: Task) -> str:
    return f"{settings.raw_config_name}--{task.name}"


load_vima_instance_from_path_fn = partial(
    load_instance_from_pickled_path,
    instance=VIMAInstance,
    load_from_path_fn=load_pickle,
)


def create_validation_split(
    dataset: datasets.Dataset,
    *,
    max_num_validation_instances: int,
    num_workers: int,
    writer_batch_size: int,
    seed: int = 0,
    stratify_column: str = "task",
) -> datasets.DatasetDict:
    """Create the validation split for the dataset.

    From the HF docs, it says that we need to flatten the indices for the dataset before shuffling,
    otherwise it takes _forever_. When calling the `train_test_split` method, it doesn't allow us
    to use a larger batch size or more workers for the indices flattening, and it then wants to
    take 24 hour to do. To try make it go faster, we flatten it ourselves so it will create the
    cache as fast as possible, so creating the splits are then faster (as they are just loading the
    cache),
    """
    dataset = dataset.flatten_indices(writer_batch_size=writer_batch_size, num_proc=num_workers)
    dataset_split = dataset.train_test_split(
        test_size=max_num_validation_instances,
        stratify_by_column=stratify_column,
        seed=seed,
        writer_batch_size=writer_batch_size,
    )
    dataset_dict = datasets.DatasetDict(
        {"train": dataset_split["train"], "valid": dataset_split["test"]}
    )
    return dataset_dict


def create_hf_dataset_for_each_task(
    parsed_instances_dir: Path,
    num_workers: int,
    writer_batch_size: int,
) -> list[datasets.Dataset]:
    """Create a HF dataset for each task.

    The dataset for each task is cached accordingly, so they can be easily loaded again if needing
    to re-run.
    """
    all_datasets: list[datasets.Dataset] = []

    for idx, task in enumerate(Task):
        data_root_for_task: Path = parsed_instances_dir.joinpath(task.name)
        # If there are no instances for that task, we can move on
        if not data_root_for_task.exists():
            continue

        instance_paths = get_pickled_instance_paths(data_root_for_task)
        logger.info(
            f"Task {idx+1}/{len(Task)}: Creating HF dataset for {task} from"
            f" {len(instance_paths)} instances..."
        )
        dataset_for_task = create_hf_dataset_from_paths(
            instance_paths,
            load_instance_from_path_fn=load_vima_instance_from_path_fn,
            dataset_features=VIMAInstance.dataset_features(),
            num_workers=num_workers,
            writer_batch_size=writer_batch_size,
            dataset_builder_kwargs={
                "dataset_name": settings.safe_hf_repo_id,
                "config_name": _get_config_name_for_task(task),
            },
        )

        all_datasets.append(dataset_for_task)

    return all_datasets


def save_dataset_for_task(
    dataset_with_split: datasets.DatasetDict,
    task_identifier: int,
    parsed_hf_dataset_dir: Path,
    num_workers: int,
    max_shard_size: str,
) -> None:
    """Save the HF dataset for just the provided task.

    Filter the main dataset across the splits and then make sure it works.
    """
    task = Task(task_identifier)

    logger.info(f"Task ID {task_identifier}: Saving dataset with task {task}...")
    task_dataset_with_split = dataset_with_split.filter(
        lambda example, task_identifier=task_identifier: example["task"] == task_identifier
    )

    # Manually update the config name for the dataset so that it includes the task name
    for dataset_split in task_dataset_with_split.values():
        dataset_split._info.config_name = _get_config_name_for_task(task)  # noqa: SLF001

    task_dataset_with_split.save_to_disk(
        parsed_hf_dataset_dir.joinpath(task.name),
        max_shard_size=max_shard_size,
        num_proc=num_workers,
    )


def create_raw_dataset_per_task(
    parsed_instances_dir: Annotated[
        Path, typer.Argument(help="Where to save all of the parsed instances")
    ] = settings.parsed_instances_dir,
    num_workers: Annotated[
        int,
        typer.Option(help="Number of workers when creating the dataset for each task."),
    ] = 1,
    writer_batch_size: Annotated[
        int,
        typer.Option(help="Batch size when creating the dataset for each task."),
    ] = 500,
) -> None:
    """Convert the parsed VIMA instances for each task into a HF dataset.

    For each task, we load the pickled files and turn them into a HF dataset.

    The reason we don't do it in a whole dataset is for two reasons:
        1. If we want to add more training tasks, then we would need to re-run the entire process,
           which can take hours, which is annoying.
        2. There were issues when saving the entire dataset in one go, and I have no idea why. So
           just do it in parts which seems to work and was way faster since we can crank the
           num-workers up without needing to worry about blowing the memory.
    """
    logger.info("Creating dataset for each task...")
    create_hf_dataset_for_each_task(
        parsed_instances_dir=parsed_instances_dir,
        num_workers=num_workers,
        writer_batch_size=writer_batch_size,
    )


def create_stratified_splits_across_tasks(
    parsed_instances_dir: Annotated[
        Path, typer.Argument(help="Where to save all of the parsed instances")
    ] = settings.parsed_instances_dir,
    parsed_hf_dataset_dir: Annotated[
        Path, typer.Argument(help="Where to save the HF datasets (for each task)")
    ] = settings.parsed_hf_dataset_dir,
    num_workers: Annotated[
        int,
        typer.Option(help="Number of workers when creating the dataset for each task."),
    ] = 1,
    writer_batch_size: Annotated[
        int,
        typer.Option(help="Batch size when creating the dataset for each task."),
    ] = 50000,
) -> None:
    """Create train-valid split across all tasks using stratified sampling.

    Since we can crank the num. workers and the writer batch size up, we should. This will make it
    go waaaaaaaaaaaay faster when creating things.

    Once the dataset for each task is loaded from the cache, concatenate them together and create
    the train-valid split using stratified sampling. Then, split back into each task and save.
    """
    logger.info("Loading dataset for each task...")
    task_datasets = create_hf_dataset_for_each_task(
        parsed_instances_dir=parsed_instances_dir,
        num_workers=num_workers,
        writer_batch_size=writer_batch_size,
    )

    logger.info("Concatenating datasets together...")
    dataset = datasets.concatenate_datasets(
        task_datasets,
        info=datasets.DatasetInfo(
            dataset_name=settings.safe_hf_repo_id, config_name=settings.raw_config_name
        ),
    )

    logger.info("Creating the train-valid split...")
    dataset_with_split = create_validation_split(
        dataset,
        max_num_validation_instances=settings.num_validation_instances,
        seed=settings.seed,
        num_workers=num_workers,
        writer_batch_size=writer_batch_size,
    )

    task_identifiers_in_dataset_per_split: dict[str, list[int]] = dataset_with_split.unique("task")
    task_identifiers_in_dataset = set(
        itertools.chain.from_iterable(task_identifiers_in_dataset_per_split.values())
    )

    logger.info("Sharding and saving dataset by task...")
    for task_identifier in task_identifiers_in_dataset:
        save_dataset_for_task(
            dataset_with_split,
            task_identifier,
            parsed_hf_dataset_dir,
            num_workers,
            settings.max_shard_size,
        )


if __name__ == "__main__":
    create_raw_dataset_per_task()
