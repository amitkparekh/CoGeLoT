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
    num_validation_instances: int,
    writer_batch_size: int,
    seed: int = 0,
) -> datasets.DatasetDict:
    """Create the validation split for the dataset."""
    dataset_split = dataset.train_test_split(
        test_size=num_validation_instances,
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
) -> dict[Task, datasets.Dataset]:
    """Create a HF dataset for each task.

    The dataset for each task is cached accordingly, so they can be easily loaded again if needing
    to re-run.
    """
    all_datasets: dict[Task, datasets.Dataset] = {}

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

        all_datasets[task] = dataset_for_task

    return all_datasets


def calculate_num_validation_instances_per_task(
    total_num_validation_instances: int,
    num_examples_per_task: dict[Task, int],
) -> dict[Task, int]:
    """Calculate the number of validation instances to use per task.

    The number of validation instances to use per task is proportional to the number of examples
    for that task.

    Yes, this is stratified sampling, but we are implementing it ourselves because HF's
    implementation was taking forever with everything else.
    """
    total_num_examples = sum(num_examples_per_task.values())
    num_validation_instances_per_task = {
        task: int(total_num_validation_instances * (num_examples / total_num_examples))
        for task, num_examples in num_examples_per_task.items()
    }
    return num_validation_instances_per_task


def create_raw_dataset_per_task(
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
    ] = 500,
) -> None:
    """Convert the parsed VIMA instances for each task into a HF dataset.

    For each task, we load the pickled files and turn them into a HF dataset. Once the dataset for
    each task is made, we create the train-valid split using stratified sampling. Then, we save
    each task separately.

    The reason we don't do it in a whole dataset is for two reasons:
        1. If we want to add more training tasks, then we would need to re-run the entire process,
           which can take hours, which is annoying.
        2. There were issues when saving the entire dataset in one go, and I have no idea why. So
           just do it in parts which seems to work and was way faster since we can crank the
           num-workers up without needing to worry about blowing the memory.

    When saving, it can take some time for the thing to start up. The larger the dataset and the
    more workers you have, the longer it takes to get going. Unfortunately, there is no way or
    knowing what it is doing or why it is taking so long, so we need to be a bit patient.    Anecdotally, when saving the dataset per task, it seems to take about 2-3 minutes to start 60
    workers. This unknown time requirement is why we don't just save the entire dataset together,
    because we have no clue what is actually happening and whether or not it has deadlocked.
    """
    logger.info("Creating dataset for each task...")
    task_datasets = create_hf_dataset_for_each_task(
        parsed_instances_dir=parsed_instances_dir,
        num_workers=num_workers,
        writer_batch_size=writer_batch_size,
    )

    logger.info("Calculate the number of validation instances to use per task...")
    num_examples_per_task = {task: len(dataset) for task, dataset in task_datasets.items()}

    num_validation_instances_per_task = calculate_num_validation_instances_per_task(
        total_num_validation_instances=settings.num_validation_instances,
        num_examples_per_task=num_examples_per_task,
    )

    logger.info("Creating the train-valid split for each task...")
    task_datasets_with_split = {
        task: create_validation_split(
            dataset,
            num_validation_instances=num_validation_instances_per_task[task],
            seed=settings.seed,
            writer_batch_size=writer_batch_size,
        )
        for task, dataset in task_datasets.items()
    }

    logger.info("Saving dataset by task...")
    for idx, (task, dataset) in enumerate(task_datasets_with_split.items()):
        logger.info(f"Task {idx+1}: Saving dataset for {task}...")
        dataset.save_to_disk(
            parsed_hf_dataset_dir.joinpath(task.name),
            num_shards={"train": 20, "valid": 5},
            max_shard_size=settings.max_shard_size,
            num_proc=num_workers,
        )


if __name__ == "__main__":
    create_raw_dataset_per_task()
