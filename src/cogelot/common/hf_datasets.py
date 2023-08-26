import os
from pathlib import Path
from typing import TypeVar

import datasets
from datasets.distributed import split_dataset_by_node
from huggingface_hub import snapshot_download


U = TypeVar("U", datasets.Dataset, datasets.IterableDataset)


def maybe_split_dataset_by_node(dataset: U) -> U:
    """Maybe split the dataset per node, if that's a thing that needs doing.

    If not, do nothing.
    """
    current_rank = os.environ.get("RANK", None)
    world_size = os.environ.get("WORLD_SIZE", None)

    if current_rank is None or world_size is None:
        return dataset

    return split_dataset_by_node(dataset, rank=int(current_rank), world_size=int(world_size))


def download_parquet_files_from_hub(
    repo_id: str, *, name: str | None = None, max_workers: int = 8
) -> None:
    """Download the parquet data files from the dataset on the hub.

    This is faster than using `datasets.load_dataset`. `datasets.load_dataset` doesn't download as
    fast as it could do. Even if we are not being rate limited, it is only downloading one SPLIT at
    a time. Not one file, not one shard, but per split.

    However, doing it this way does not automatically fill the cache, so you cannot use
    `load_dataset` when loading the dataset. The `load_dataset_from_parquet_files` function (below)
    is there to load the dataset from the parquet files and returns the `DatasetDict`.

    If providing the `name`, then only the parquet files within that directory will be downloaded.
    If no name is provided, then we just download all the parquet files.
    """
    pattern = "*.parquet"
    library_name = repo_id
    if name:
        pattern = f"**{name}/*.parquet"
        library_name = f"{repo_id}/{name}"

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=None,
        library_name=library_name,
        allow_patterns=pattern,
        max_workers=max_workers,
        resume_download=True,
    )


def get_location_of_parquet_files(repo_id: str) -> Path:
    """Get the path to the location where the parquet files were downloaded."""
    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            library_name=repo_id,
            local_files_only=True,
            local_dir=None,
        )
    )


def load_dataset_from_parquet_files(
    data_dir: Path,
    *,
    num_proc: int | None = None,
    dataset_splits: tuple[str, ...] = ("train", "valid"),
) -> datasets.DatasetDict:
    """Load the dataset from the parquet files."""
    # Get the parquet files per split
    parquet_files_per_split = {
        split: list(map(str, data_dir.rglob(f"{split}*.parquet"))) for split in dataset_splits
    }

    # Load the dataset with the splits
    dataset_dict = datasets.load_dataset(
        "parquet", data_files=parquet_files_per_split, num_proc=num_proc
    )
    assert isinstance(dataset_dict, datasets.DatasetDict)
    return dataset_dict
