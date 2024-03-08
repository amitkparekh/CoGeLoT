from pathlib import Path
from typing import Annotated, Any

import typer
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from cogelot.common.settings import Settings
from cogelot.entrypoints.preprocess_instances import load_parsed_datasets_for_each_task
from cogelot.structures.vima import VIMAInstance

settings = Settings()


def _convert_instance_to_metadata(raw_instance: dict[str, Any]) -> str:
    """Convert the instance path to metadata.

    Since we use this as a "with_transform" function, we need to expand every list within the
    thing.
    """
    return VIMAInstance.model_validate(raw_instance).to_metadata().model_dump_json()


def dump_dataset_metadata(  # noqa: WPS210
    parsed_hf_dataset_dir: Annotated[
        Path, typer.Argument(help="Where to get the parsed HF datasets (for each task)")
    ] = settings.parsed_hf_dataset_dir,
    train_metadata_output_file: Annotated[
        Path, typer.Argument(help="Where to save the training metadata")
    ] = settings.train_instances_metadata,
    valid_metadata_output_file: Annotated[
        Path, typer.Argument(help="Where to save the validation metadata")
    ] = settings.valid_instances_metadata,
    *,
    num_workers: Annotated[int, typer.Option(help="Number of workers.")] = 1,
) -> None:
    """Dump all the dataset metadata."""
    logger.info("Loading the parsed datasets for each task...")
    parsed_datasets_per_task_iterator = load_parsed_datasets_for_each_task(parsed_hf_dataset_dir)

    train_metadata = []
    valid_metadata = []

    for task, dataset in parsed_datasets_per_task_iterator:
        logger.info(f"Processing the {task} instances...")

        logger.debug("Loading datasets with transforms")
        train_dataset = dataset["train"].with_transform(_convert_instance_to_metadata)
        valid_dataset = dataset["valid"].with_transform(_convert_instance_to_metadata)

        logger.debug("Creating dataloaders")
        train_dataloader = DataLoader(
            train_dataset,  # pyright: ignore[reportArgumentType]
            batch_size=None,
            num_workers=num_workers,
            batch_sampler=None,
        )
        valid_dataloader = DataLoader(
            valid_dataset,  # pyright: ignore[reportArgumentType]
            batch_size=None,
            num_workers=num_workers,
            batch_sampler=None,
        )

        logger.debug("Processing train instances")
        with tqdm(f"{task}: Train instances", total=len(train_dataset)) as train_progress_bar:
            for train_instance in train_dataloader:
                train_metadata.append(train_instance)
                train_progress_bar.update(1)

        logger.debug("Processing valid instances")
        with tqdm(f"{task}: Valid instances", total=len(valid_dataset)) as valid_progress_bar:
            for valid_instance in valid_dataloader:
                valid_metadata.append(valid_instance)
                valid_progress_bar.update(1)

    logger.info("Writing to files")
    train_metadata_output_file.write_text("\n".join(train_metadata))
    valid_metadata_output_file.write_text("\n".join(valid_metadata))


if __name__ == "__main__":
    typer.run(dump_dataset_metadata)
