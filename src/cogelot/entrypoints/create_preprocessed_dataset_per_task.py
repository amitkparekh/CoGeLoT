from functools import partial
from pathlib import Path
from typing import Annotated

import datasets
import typer
from loguru import logger
from rich.progress import track

from cogelot.common.io import load_pickle
from cogelot.data.datasets import create_hf_dataset_from_paths, load_instance_from_pickled_path
from cogelot.entrypoints.settings import Settings
from cogelot.structures.model import PreprocessedInstance
from cogelot.structures.vima import Task


settings = Settings()


def _get_config_name_for_task(task: Task) -> str:
    return f"{settings.preprocessed_config_name}--{task.name}"


def _get_preprocessed_instances_dir_per_task(preprocessed_instances_dir: Path) -> dict[Task, Path]:
    """Get the preprocessed instances directory for each task."""
    preprocessed_instances_dir_per_task: dict[Task, Path] = {
        Task[task_dir.name]: task_dir for task_dir in preprocessed_instances_dir.iterdir()
    }
    return preprocessed_instances_dir_per_task


def get_all_preprocessed_instance_paths(preprocessed_instance_dir: Path) -> list[Path]:
    """Get all the parsed VIMA instance paths."""
    path_iterator = preprocessed_instance_dir.rglob("*.pkl.gz")
    return list(track(path_iterator, description="Getting all preprocessed instance paths"))


load_preprocessed_instance_from_path_fn = partial(
    load_instance_from_pickled_path, instance=PreprocessedInstance, load_from_path_fn=load_pickle
)


def create_preprocessed_hf_dataset_for_task(
    *,
    task: Task,
    task_instances_dir: Path,
    output_dir: Path,
    num_workers: int,
    dataset_name: str,
    writer_batch_size: int,
    max_shard_size: str,
) -> None:
    """Convert the preprocessed instances into a HF dataset for the given task."""
    logger.info(f"Get all the preprocessed instance paths for {task}...")
    all_train_instance_paths = get_all_preprocessed_instance_paths(
        task_instances_dir.joinpath("train/")
    )
    all_valid_instance_paths = get_all_preprocessed_instance_paths(
        task_instances_dir.joinpath("valid/")
    )

    logger.info("Creating the HF dataset for each split...")
    preprocessed_train_dataset = create_hf_dataset_from_paths(
        paths=all_train_instance_paths,
        load_instance_from_path_fn=load_preprocessed_instance_from_path_fn,
        dataset_features=PreprocessedInstance.dataset_features(),
        num_workers=num_workers,
        writer_batch_size=writer_batch_size,
        dataset_builder_kwargs={
            "dataset_name": dataset_name,
            "config_name": _get_config_name_for_task(task),
        },
    )
    preprocessed_valid_dataset = create_hf_dataset_from_paths(
        paths=all_valid_instance_paths,
        load_instance_from_path_fn=load_preprocessed_instance_from_path_fn,
        dataset_features=PreprocessedInstance.dataset_features(),
        num_workers=num_workers,
        writer_batch_size=writer_batch_size,
        dataset_builder_kwargs={
            "dataset_name": dataset_name,
            "config_name": _get_config_name_for_task(task),
        },
    )
    # Merge the two into a dataset dict
    dataset_dict = datasets.DatasetDict(
        {"train": preprocessed_train_dataset, "valid": preprocessed_valid_dataset}
    )

    logger.info(f"Saving HF dataset for {task} to disk...")
    dataset_dict.save_to_disk(
        dataset_dict_path=str(output_dir.resolve()),
        max_shard_size=max_shard_size,
        num_proc=num_workers,
    )


def create_preprocessed_dataset_per_task(
    preprocessed_instances_dir: Annotated[
        Path, typer.Option(help="Directory to save the preprocessed instances to.")
    ] = settings.preprocessed_data_dir,
    preprocessed_hf_dataset_dir: Annotated[
        Path, typer.Option(help="Directory to save the preprocessed HF dataset.")
    ] = settings.preprocessed_hf_dataset_dir,
    num_workers: Annotated[int, typer.Option(help="Number of workers.")] = 1,
) -> None:
    """Create a HF dataset for the preprocessed instances for each task.

    Similar to the parsed instances, we do this for each task separately. This is because things
    errored and took way too long when doing things all together.
    """
    logger.info("Getting the instance dir for each task...")
    instance_dir_per_task = _get_preprocessed_instances_dir_per_task(preprocessed_instances_dir)

    # Create the HF dataset for each task and save to disk
    logger.info("Creating the preprocessed HF dataset for each task...")
    for task, task_instances_dir in instance_dir_per_task.items():
        create_preprocessed_hf_dataset_for_task(
            task=task,
            task_instances_dir=task_instances_dir,
            output_dir=preprocessed_hf_dataset_dir.joinpath(task.name),
            num_workers=num_workers,
            dataset_name=settings.safe_hf_repo_id,
            writer_batch_size=settings.writer_batch_size,
            max_shard_size=settings.max_shard_size,
        )


if __name__ == "__main__":
    create_preprocessed_dataset_per_task()
