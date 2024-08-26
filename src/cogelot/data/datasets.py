from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any, cast

import datasets
from loguru import logger
from pydantic import BaseModel


def load_instance_from_path(
    path: Path, *, load_from_path_fn: Callable[[Path], Any], instance: type[BaseModel]
) -> dict[str, Any]:
    """Load the instance from the pickled path."""
    try:
        return instance.model_validate(load_from_path_fn(path)).model_dump()
    except Exception as err:  # noqa:  BLE001
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
    dataset_builder_kwargs: dict[str, Any] | None = None,
) -> datasets.Dataset:
    """Create HF dataset from instance paths."""
    dataset_builder_kwargs = dataset_builder_kwargs or {}

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
            **dataset_builder_kwargs,
        ),
    )
    return hf_dataset


def only_select_indices_within_range(
    dataset: datasets.Dataset, *, start: int, end: int
) -> datasets.Dataset:
    """Only select indices within the range."""
    dataset = dataset.select(range(start, end))
    return dataset


def repeat_dataset_for_batch_size(
    dataset: datasets.Dataset, *, batch_size: int
) -> datasets.Dataset:
    """Repeat the dataset to match the batch size."""
    dataset = datasets.concatenate_datasets([dataset for _ in range(batch_size)])
    return dataset
