from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, cast

import datasets
import psutil
import typer
from loguru import logger
from rich.progress import track

from cogelot.data.parse import (
    create_vima_instance_from_instance_dir,
    get_all_raw_instance_directories,
)
from cogelot.structures.vima import VIMAInstance


NUM_VALIDATION_INSTANCES = 50000
WRITER_BATCH_SIZE = 10000
STORAGE_DATA_DIR = Path("storage/data/")
RAW_DATA_DIR = STORAGE_DATA_DIR.joinpath("raw/vima_v6/")


def is_requested_more_workers_than_physical_cores(num_workers: int) -> bool:
    """Check if the number of workers requested is greater than the number of physical cores."""
    physical_core_count = psutil.cpu_count(logical=False)
    return num_workers > physical_core_count


def get_raw_instance_directories(raw_data_root: Path) -> list[Path]:
    """Get all the raw instance directories."""
    path_iterator = get_all_raw_instance_directories(raw_data_root)
    all_paths = list(track(path_iterator, description="Getting all raw instance paths"))
    return all_paths


def parse_vima_instances_from_path(paths: list[Path]) -> Iterator[dict[str, Any]]:
    """Yield VIMA instances from paths for the HF dataset."""
    yield from (create_vima_instance_from_instance_dir(path).model_dump() for path in paths)


def create_hf_dataset_from_vima_instance_paths(
    paths: list[Path], *, num_workers: int
) -> datasets.Dataset:
    """Create HF dataset from VIMA instance paths."""
    hf_dataset = cast(
        datasets.Dataset,
        datasets.Dataset.from_generator(
            parse_vima_instances_from_path,
            features=VIMAInstance.dataset_features(),
            num_proc=num_workers if num_workers else None,
            gen_kwargs={"paths": paths},
            writer_batch_size=WRITER_BATCH_SIZE,
        ),
    )
    return hf_dataset


def create_validation_split(
    dataset: datasets.Dataset,
    *,
    max_num_validation_instances: int,
    seed: int = 0,
    writer_batch_size: int = WRITER_BATCH_SIZE,
    stratify_column: str = "task",
) -> datasets.DatasetDict:
    """Create the validation split for the dataset."""
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


def convert_raw_dataset_to_hf(
    raw_data_root: Annotated[
        Path, typer.Argument(help="Root directory for the raw data")
    ] = RAW_DATA_DIR,
    num_workers: Annotated[int, typer.Option(help="Number of workers.")] = 2,
    num_validation_instances: Annotated[
        int,
        typer.Option(
            help="Maximum number of validation instances, created using stratefied sampling"
        ),
    ] = NUM_VALIDATION_INSTANCES,
    hf_repo_id: Annotated[str, typer.Option(help="Repository ID on HF")] = "amitkparekh/vima",
    max_shard_size: Annotated[
        str,
        typer.Option(help="Maximum shard size for the dataset."),
    ] = "1GB",
    seed: Annotated[int, typer.Option(help="Seed for the stratified sampling.")] = 1000,
) -> None:
    """Convert the raw dataset to a dataset on HF.

    That means parsing, validating, constructing the thing, and then uploading it to HF.

    Creating each VIMA instance from the raw data is an intensive task, but also a slow one that is
    bottlenecked by the IOPS of your drive. Increasing the number of workers will help with this to
    a point, but there will be a limit to this. You're just going to have to wait. During this,
    watch the memory usage too. Too many workers will cause the system to run out of memory and
    likely crash the running.
    """
    all_raw_instance_paths = get_raw_instance_directories(raw_data_root)

    if is_requested_more_workers_than_physical_cores(num_workers):
        logger.warning(
            "You are requesting more workers than physical cores. This is not recommended. You"
            " should use the number of physical cores in the system (so whatever it says in htop /"
            " 2). If you don't heed this warning, then creation of the dataset will likely fail —"
            " well that's what I've seen anyway."
        )

    logger.info("Creating the HF dataset...")
    dataset = create_hf_dataset_from_vima_instance_paths(
        all_raw_instance_paths, num_workers=num_workers
    )

    logger.info("Creating the train-valid split...")
    dataset_with_split = create_validation_split(
        dataset, max_num_validation_instances=num_validation_instances, seed=seed
    )

    logger.info("Pushing the dataset to the hub...")
    logger.info(
        "This will take a while. It might look like it's doing nothing, but it is taking a while."
    )
    dataset_with_split.push_to_hub(hf_repo_id, max_shard_size=max_shard_size)


if __name__ == "__main__":
    convert_raw_dataset_to_hf()
