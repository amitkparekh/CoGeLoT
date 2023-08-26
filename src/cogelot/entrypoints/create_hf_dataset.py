from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Any, cast

import datasets
import typer
from loguru import logger
from rich.progress import track

from cogelot.common.io import load_pickle
from cogelot.entrypoints.settings import Settings
from cogelot.structures.vima import VIMAInstance


settings = Settings()


def get_all_parsed_vima_instances_paths(parsed_data_dir: Path) -> list[Path]:
    """Get all the parsed VIMA instance paths."""
    path_iterator = parsed_data_dir.rglob("*.pkl.gz")
    return list(track(path_iterator, description="Getting all parsed VIMA instance paths"))


def yield_vima_instance_from_path(path: Path) -> VIMAInstance:
    """Load and yield the VIMA instance from the path."""
    return VIMAInstance.model_validate(load_pickle(path))


def yield_vima_instances_for_hf_generator(
    paths: list[Path], num_workers: int
) -> Iterator[dict[str, Any]]:
    """Yield the VIMA instances for the HF generator.

    We return the model_dump because that's what the HF dataset wants.
    """
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(yield_vima_instance_from_path, path) for path in paths]
        yield from (future.result().model_dump() for future in as_completed(futures))


def create_hf_dataset_from_vima_instance_paths(
    paths: list[Path], *, num_workers: int
) -> datasets.Dataset:
    """Create HF dataset from VIMA instance paths."""
    hf_dataset = cast(
        datasets.Dataset,
        datasets.Dataset.from_generator(
            yield_vima_instances_for_hf_generator,
            features=VIMAInstance.dataset_features(),
            gen_kwargs={"paths": paths, "max_workers": num_workers},
            writer_batch_size=settings.writer_batch_size,
        ),
    )
    return hf_dataset


def create_validation_split(
    dataset: datasets.Dataset,
    *,
    max_num_validation_instances: int,
    seed: int = 0,
    writer_batch_size: int = settings.writer_batch_size,
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


def convert_vima_instance_to_hf_dataset(
    parsed_data_root: Annotated[
        Path, typer.Argument(help="Root directory for the parsed VIMA instances")
    ] = settings.parsed_data_dir,
    num_workers: Annotated[int, typer.Option(help="Number of workers.")] = 2,
    num_validation_instances: Annotated[
        int,
        typer.Option(
            help="Maximum number of validation instances, created using stratefied sampling"
        ),
    ] = settings.num_validation_instances,
    hf_repo_id: Annotated[
        str, typer.Option("Repository ID for the dataset on HF")
    ] = settings.hf_repo_id,
    max_shard_size: Annotated[
        str, typer.Option("Maximum shard size for the dataset")
    ] = settings.max_shard_size,
    seed: Annotated[int, typer.Option(help="Seed for the stratified sampling.")] = 1000,
) -> None:
    """Convert the parsed VIMA instances to a dataset on HF.

    We load all the pickled files, parse them to ensure they're correct, turn them into the HF
    dataset, and upload it to HF.
    """
    all_instance_paths = get_all_parsed_vima_instances_paths(parsed_data_root)

    logger.info("Creating the HF dataset...")
    dataset = create_hf_dataset_from_vima_instance_paths(
        all_instance_paths, num_workers=num_workers
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
    convert_vima_instance_to_hf_dataset()
