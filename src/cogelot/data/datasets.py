from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any, TypeVar, cast

import datasets
from loguru import logger
from pydantic import BaseModel
from rich.progress import track

from cogelot.structures.vima import Task


T = TypeVar("T", datasets.Dataset, datasets.DatasetDict)


def set_dataset_format(dataset: T) -> T:
    """Set dataset format for VIMA instances."""
    columns_with_tensors = ["word_batch", "image_batch", "observations", "actions"]
    dataset = dataset.with_format("torch", columns=columns_with_tensors, output_all_columns=True)
    return dataset


def load_instance_from_pickled_path(
    path: Path, *, load_from_path_fn: Callable[[Path], Any], instance: type[BaseModel]
) -> dict[str, Any]:
    """Load the instance from the pickled path."""
    try:
        return instance.model_validate(load_from_path_fn(path)).model_dump()
    except Exception as err:
        logger.exception(f"Something went wrong with {path}")
        raise err from None


def _yield_instances_for_hf_generator(
    paths: list[Path], *, load_instance_from_path_fn: Callable[[Path], dict[str, Any]]
) -> Iterator[dict[str, Any]]:
    """Generator to yield instances from paths to feed into the HF dataset generator."""
    yield from map(load_instance_from_path_fn, paths)


def create_hf_dataset_from_paths(
    paths: list[Path],
    *,
    load_instance_from_path_fn: Callable[[Path], dict[str, Any]],
    dataset_features: datasets.Features,
    num_workers: int,
    writer_batch_size: int | None,
) -> datasets.Dataset:
    """Create HF dataset from VIMA instance paths."""
    yield_instance_for_generator_fn = partial(
        _yield_instances_for_hf_generator, load_instance_from_path_fn=load_instance_from_path_fn
    )
    hf_dataset = cast(
        datasets.Dataset,
        datasets.Dataset.from_generator(
            yield_instance_for_generator_fn,
            features=dataset_features,
            gen_kwargs={"paths": paths},
            num_proc=max(num_workers, 1),
            writer_batch_size=writer_batch_size,
        ),
    )
    return hf_dataset


def get_pickled_instance_paths(root_dir: Path) -> list[Path]:
    """Get all the pickled instances from the root dir."""
    path_iterator = root_dir.rglob("*.pkl.gz")
    return list(track(path_iterator, description=f"Getting pickled paths from {root_dir}"))


def create_hf_dataset_from_pickled_instances(
    pickled_instance_root: Path,
    *,
    temp_output_dir: Path,
    num_workers: int,
    max_shard_size: str,
    writer_batch_size: int,
    dataset_features: datasets.Features,
    load_instance_from_path_fn: Callable[[Path], dict[str, Any]],
) -> datasets.Dataset:
    """Create a HF dataset by creating one for each task and then merging them together.

    There was some weird phantom error when trying to run this creation on all the tasks at once. I
    don't understand why but when I was running one task at a time to debug it, it worked fine. So,
    that's what we're going to do here.
    """
    for task in Task:
        data_root_for_task: Path = pickled_instance_root.joinpath(task.name)
        # If there are no instances for that task, we can move on
        if not data_root_for_task.exists():
            continue

        output_dir_for_task: Path = temp_output_dir.joinpath(task.name)

        instance_paths = get_pickled_instance_paths(data_root_for_task)
        logger.info(f"Creating HF dataset for {task} from {len(instance_paths)} instances...")
        dataset_for_task = create_hf_dataset_from_paths(
            instance_paths,
            load_instance_from_path_fn=load_instance_from_path_fn,
            dataset_features=dataset_features,
            num_workers=num_workers,
            writer_batch_size=writer_batch_size,
        )

        logger.info(f"Saving dataset to {temp_output_dir}")
        dataset_for_task.save_to_disk(
            output_dir_for_task,
            num_proc=num_workers,
            max_shard_size=max_shard_size,
            storage_options={"writer_batch_size": writer_batch_size},
        )

    logger.info("Gathering arrow files from all tasks...")
    all_arrow_files = list(map(str, temp_output_dir.rglob("*.arrow")))

    logger.info(f"Loading dataset from {len(all_arrow_files)} files...")
    dataset = datasets.load_dataset(
        "arrow",
        data_files=all_arrow_files,
        features=dataset_features,
        num_proc=num_workers,
    )
    if isinstance(dataset, datasets.DatasetDict):
        dataset = dataset["train"]

    assert isinstance(dataset, datasets.Dataset)
    return dataset
