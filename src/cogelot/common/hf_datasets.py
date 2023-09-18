import os
from pathlib import Path
from typing import Literal, TypeVar

import datasets
from datasets.distributed import split_dataset_by_node
from huggingface_hub import snapshot_download
from loguru import logger


SavedFileExtension = Literal["parquet", "arrow"]

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
    repo_id: str, *, config_name_prefix: str | None = None, max_workers: int = 8
) -> None:
    """Download the parquet data files from the dataset on the hub.

    This is faster than using `datasets.load_dataset`. `datasets.load_dataset` doesn't download as
    fast as it could do. Even if we are not being rate limited, it is only downloading one SPLIT at
    a time. Not one file, not one shard, but per split.

    If providing the `config_name_prefix`, then only the parquet files for that subset is
    downloaded. If no `config_name_prefix` is provided, then we just download all the parquet
    files.
    """
    pattern = "*.parquet"
    if config_name_prefix:
        pattern = f"**{config_name_prefix}*/*.parquet"

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=None,
        allow_patterns=pattern,
        max_workers=max_workers,
        resume_download=True,
    )


def get_location_of_hub_parquet_files(repo_id: str) -> Path:
    """Get the path to the location where the parquet files were downloaded from the Hub."""
    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_files_only=True,
            local_dir=None,
        )
    )


def load_dataset_from_files(
    data_dir: Path,
    *,
    extension: SavedFileExtension,
    num_proc: int | None = None,
) -> datasets.DatasetDict:
    """Load the dataset from the parquet files."""
    file_paths = list(data_dir.rglob(f"*.{extension}"))

    # Determine the splits that we are loading
    # Each parquet file is in the form: `SPLIT-NNNNN-of-SSSSS-UUID.parquet`
    data_splits = {path.name.split("-")[0] for path in file_paths}

    # Get the possible data splits from the parquet files
    files_per_split = {
        split: list(map(str, data_dir.rglob(f"{split}*.{extension}"))) for split in data_splits
    }

    dataset_dict = datasets.load_dataset(
        extension,
        data_files=files_per_split,
        num_proc=num_proc,
    )
    assert isinstance(dataset_dict, datasets.DatasetDict)
    return dataset_dict


def load_dataset_from_disk(
    data_dir: Path, *, extension: SavedFileExtension, config_name: str, num_proc: int | None = None
) -> datasets.DatasetDict:
    """Load the dataset from disk.

    In doing so, this merges all the tasks together too. If more control is needed with which tasks
    are loaded, that can be looked at later.
    """
    # The root data dir will have subdirs for each task, prefixed by the `config_name` of the
    # dataset since that's how they were made. e.g. `{data_dir}/raw--visual_manipulation/`
    task_dataset_dirs = data_dir.glob(f"{config_name}*/")

    # Create a generator that will load the dataset for each and every task
    dataset_per_task = (
        load_dataset_from_files(task_dataset_dir, extension=extension, num_proc=num_proc)
        for task_dataset_dir in task_dataset_dirs
    )

    # Merge the splits per task together. Since there is no guarantee on the splits per task, this
    # is the best way I can think to do it. While we are at it, we merge all the dataset info's
    # together.
    collated_dataset_splits: dict[str, list[datasets.Dataset]] = {}
    dataset_info = datasets.DatasetInfo()
    for task_dataset in dataset_per_task:
        for split, dataset_split in task_dataset.items():
            dataset_info.update(dataset_split.info)
            try:
                collated_dataset_splits[split].append(dataset_split)
            except KeyError:
                collated_dataset_splits[split] = [dataset_split]

    # Force overwrite the config name for the dataset so that it's what we want it to be because
    # that's just nicer y'know?
    dataset_info.config_name = config_name

    # Merge all the tasks together by concatenating the datasets for each split
    dataset_dict = datasets.DatasetDict(
        {
            split_name: datasets.concatenate_datasets(splits, info=dataset_info)
            for split_name, splits in collated_dataset_splits.items()
        }
    )

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

    # Get the config name for the dataset
    config_name = next(iter(dataset_dict.values())).info.config_name

    logger.info(f"Pushing dataset ({config_name}) to the hub...")
    dataset_dict.push_to_hub(hf_repo_id, config_name=config_name, num_shards=num_shards)
