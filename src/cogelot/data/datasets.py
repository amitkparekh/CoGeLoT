from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any, TypeVar, cast

import datasets
from loguru import logger
from pydantic import BaseModel


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
