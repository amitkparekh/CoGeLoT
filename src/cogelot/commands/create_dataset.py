from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, cast

import datasets
import typer
from rich.progress import track

from cogelot.data.parse import (
    create_vima_instance_from_instance_dir,
    get_all_raw_instance_directories,
)
from cogelot.structures.vima import VIMAInstance


STORAGE_DATA_DIR = Path("storage/data/")
RAW_DATA_DIR = STORAGE_DATA_DIR.joinpath("raw/vima_v6/")

app = typer.Typer(add_completion=False)


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
            num_proc=num_workers,
            gen_kwargs={"paths": paths},
        ),
    )
    return hf_dataset


def create_validation_split(
    dataset: datasets.Dataset,
    *,
    max_num_validation_instances: int,
    seed: int = 0,
    writer_batch_size: int = 1000,
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


@app.command("create-hf-dataset")
def convert_raw_dataset_to_hf(
    raw_data_root: Annotated[
        Path, typer.Argument(help="Root directory for the raw data")
    ] = RAW_DATA_DIR,
    num_workers: Annotated[int, typer.Option(help="Number of workers.")] = 1,
    num_validation_instances: Annotated[
        int,
        typer.Option(
            help="Maximum number of validation instances, created using stratefied sampling"
        ),
    ] = 50000,
    hf_repo_id: Annotated[str, typer.Option(help="Repository ID on HF")] = "amitkparekh/vima",
    max_shard_size: Annotated[
        str,
        typer.Option(help="Maximum shard size for the dataset."),
    ] = "1GB",
    seed: Annotated[int, typer.Option(help="Seed for the stratified sampling.")] = 1000,
) -> None:
    """Convert the raw dataset to a HF dataset.

    That means parsing, validating, constructing the thing, and then uploading it to HF.
    """
    all_raw_instance_paths = get_raw_instance_directories(raw_data_root)
    dataset = create_hf_dataset_from_vima_instance_paths(
        all_raw_instance_paths, num_workers=num_workers
    )
    dataset_with_split = create_validation_split(
        dataset, max_num_validation_instances=num_validation_instances, seed=seed
    )
    dataset_with_split.push_to_hub(hf_repo_id, max_shard_size=max_shard_size)


if __name__ == "__main__":
    app()
