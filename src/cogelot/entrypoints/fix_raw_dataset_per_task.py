from pathlib import Path
from typing import Annotated, cast

import datasets
import torch
import typer
from loguru import logger

from cogelot.common.settings import Settings
from cogelot.data.datasets import (
    create_hf_dataset_from_paths,
)
from cogelot.entrypoints.create_raw_dataset_per_task import (
    load_vima_instance_from_path_fn,
)
from cogelot.entrypoints.preprocess_instances import load_parsed_datasets_for_each_task
from cogelot.structures.vima import VIMAInstance

settings = Settings()


def fix_raw_dataset_per_task(  # noqa: WPS210
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
    """Recreate the raw dataset for each task.

    I created the datasets forgot to include some key information to make transforming object
    placeholders to natural language possible. So I need to re-create every single instance in the
    dataset without ruining the train-valid split. Thankfully, it appears that running the
    train-valid split on a dataset in HF is deterministic when there is a seed, which is what I
    did. However, it's a bunch to run and I don't really want to put all that time in and risk
    something going wrong, SO WE PATCH IT!

    This has primarily been taken from `cogelot.entrypoints.create_raw_dataset_per_task`.

    Also, running this assumes that all the instances have been re-parsed.
    """
    logger.info("Loading the dataset(s)...")
    dataset_per_task_iterator = load_parsed_datasets_for_each_task(parsed_hf_dataset_dir)

    for task, dataset in dataset_per_task_iterator:
        data_root_for_task: Path = parsed_instances_dir.joinpath(task.name)

        # Get all the indices within the train and valid split separately.
        logger.info(f"Getting indices for {task}...")
        train_indices: list[int] = cast(torch.Tensor, dataset["train"]["index"]).flatten().tolist()
        valid_indices: list[int] = cast(torch.Tensor, dataset["valid"]["index"]).flatten().tolist()

        # Because the index for each instance is just the index in the raw path, we can use that to
        # get the instance path for each instance
        train_instance_paths: list[Path] = [
            data_root_for_task.joinpath(f"{task.name}_{idx}.pkl.gz") for idx in train_indices
        ]
        valid_instance_paths: list[Path] = [
            data_root_for_task.joinpath(f"{task.name}_{idx}.pkl.gz") for idx in valid_indices
        ]

        logger.info(f"Task {task}: Got {len(train_instance_paths)} train instances to fix")
        train_instances = create_hf_dataset_from_paths(
            train_instance_paths,
            load_instance_from_path_fn=load_vima_instance_from_path_fn,
            dataset_features=VIMAInstance.dataset_features(),
            num_workers=num_workers,
            writer_batch_size=writer_batch_size,
            dataset_builder_kwargs={
                "dataset_name": settings.safe_hf_repo_id,
                "config_name": settings.get_config_name_for_task(task, stage="parsing"),
            },
        )

        logger.info(f"Task {task}: Got {len(train_instance_paths)} valid instances to fix")
        valid_instances = create_hf_dataset_from_paths(
            valid_instance_paths,
            load_instance_from_path_fn=load_vima_instance_from_path_fn,
            dataset_features=VIMAInstance.dataset_features(),
            num_workers=num_workers,
            writer_batch_size=writer_batch_size,
            dataset_builder_kwargs={
                "dataset_name": settings.safe_hf_repo_id,
                "config_name": settings.get_config_name_for_task(task, stage="parsing"),
            },
        )

        new_dataset = datasets.DatasetDict({"train": train_instances, "valid": valid_instances})
        logger.info(f"Saving dataset for {task}...")
        new_dataset.save_to_disk(
            parsed_hf_dataset_dir.joinpath(task.name),
            num_shards=settings.num_shards,
            num_proc=num_workers,
        )
