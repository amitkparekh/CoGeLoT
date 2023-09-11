import os
from pathlib import Path
from typing import TypeVar

import datasets
from datasets.distributed import split_dataset_by_node
from huggingface_hub import snapshot_download
from loguru import logger


T = TypeVar("T", datasets.Dataset, datasets.IterableDataset)


def maybe_split_dataset_by_node(dataset: T) -> T:
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
    if name:
        pattern = f"**{name}/*.parquet"

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=None,
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
            local_files_only=True,
            local_dir=None,
        )
    )


def load_dataset_from_parquet_files(
    data_dir: Path,
    *,
    name: str | None = None,
    num_proc: int | None = None,
    split: str | None = None,
) -> datasets.DatasetDict:
    """Load the dataset from the parquet files."""
    if name:
        data_dir = data_dir.joinpath(name)

    parquet_files = list(data_dir.rglob("*.parquet"))

    # Determine the splits that we are loading, if not specified
    if split:
        data_splits = {split}
    else:
        # Each parquet file is in the form: `SPLIT-NNNNN-of-SSSSS-UUID.parquet`
        data_splits = {parquet_file.name.split("-")[0] for parquet_file in parquet_files}

    # Get the possible data splits from the parquet files
    parquet_files_per_split = {
        split: list(map(str, data_dir.rglob(f"{split}*.parquet"))) for split in data_splits
    }

    dataset_dict = datasets.load_dataset(
        "parquet",
        name=name,
        data_files=parquet_files_per_split,
        num_proc=num_proc,
    )
    assert isinstance(dataset_dict, datasets.DatasetDict)
    return dataset_dict


def upload_dataset_to_hub(
    saved_hf_dataset_dir: Path,
    *,
    hf_repo_id: str,
    num_shards: dict[str, int],
) -> None:
    """Upload the dataset to the hub."""
    logger.info("Load dataset from disk...")
    dataset_dict = datasets.load_from_disk(str(saved_hf_dataset_dir))
    assert isinstance(dataset_dict, datasets.DatasetDict)

    logger.info("Pushing the preprocessed dataset to the hub...")
    dataset_dict.push_to_hub(hf_repo_id, num_shards=num_shards)
