import logging
import os
from collections.abc import Iterable, Iterator
from contextlib import suppress
from pathlib import Path
from typing import Literal, TypeVar, cast

import datasets
from datasets.config import HF_ENDPOINT
from datasets.distributed import split_dataset_by_node
from datasets.features.features import require_decoding
from datasets.table import embed_table_storage
from huggingface_hub import HfApi, HfFolder, snapshot_download
from huggingface_hub.utils._errors import (  # noqa: WPS436  # noqa: WPS436
    BadRequestError,
    HfHubHTTPError,
)
from loguru import logger
from tqdm import tqdm


SHARD_FILE_NAME_TEMPLATE = "{split}-{index:05d}-of-{num_shards:05d}-{fingerprint}.parquet"


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
    # Try to load each task dataset using `datasets.load_from_disk` first to get the benefit of the
    # cache. However, if that fails, then we need to load the dataset from the files directly.
    # Additionally, merge the splits per task together. Since there is no guarantee on the splits
    # per task, this is the best way I can think to do it. While we are at it, we merge all the
    # dataset info's together.
    collated_dataset_splits: dict[str, list[datasets.Dataset]] = {}
    dataset_info = datasets.DatasetInfo()
    for task_dataset_dir in task_dataset_dirs:
        try:
            task_dataset = cast(
                datasets.DatasetDict, datasets.load_from_disk(str(task_dataset_dir))
            )
        except FileNotFoundError:
            task_dataset = load_dataset_from_files(
                task_dataset_dir, extension=extension, num_proc=num_proc
            )

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


def _embed_external_files_in_shards(
    shards: Iterable[datasets.Dataset],
) -> Iterator[datasets.Dataset]:
    """Embed the external files in the shards, as done by HF.

    This is taken from `datasets.Dataset._push_parquet_files_to_hub()`, and reformatted slightly.
    """
    for shard in shards:
        shard_with_embedded_files = (
            shard.with_format("arrow")
            .map(
                embed_table_storage,
                batched=True,
                batch_size=1000,
                keep_in_memory=True,
            )
            .with_format(**shard.format)  # pyright: ignore[reportGeneralTypeIssues]
        )
        yield shard_with_embedded_files


def _shard_dataset(
    *,
    dataset: datasets.Dataset,
    num_shards: int,
    embed_external_files: bool = True,
) -> Iterator[datasets.Dataset]:
    """Shard the dataset into multiple shards."""
    shards = (dataset.shard(num_shards=num_shards, index=index) for index in range(num_shards))

    # Check if it has decodable columns
    # Find decodable columns, because if there are any, we need to:
    # embed the bytes from the files in the shards
    assert dataset.info.features is not None
    decodable_columns: list[str] = (
        [
            column_name
            for column_name, column_feature in dataset.info.features.items()
            if require_decoding(column_feature, ignore_decode_attribute=True)
        ]
        if embed_external_files
        else []
    )
    if decodable_columns:
        shards = _embed_external_files_in_shards(shards)

    return shards


def _delete_shards_with_different_fingerprint(
    *,
    output_dir: Path,
    split_name: str,
    index: int,
    num_shards: int,
    fingerprint: str,
) -> None:
    """Delete shards with a different fingerprint."""
    other_shards_for_same_index = _get_other_shard_paths_for_same_index(
        output_dir=output_dir,
        split_name=split_name,
        index=index,
        num_shards=num_shards,
        fingerprint=fingerprint,
    )

    if other_shards_for_same_index:
        for shard_path in other_shards_for_same_index:
            shard_path.unlink()
            logger.info(f"Deleted parquet file: `{shard_path}`")


def _get_other_shard_paths_for_same_index(
    *,
    output_dir: Path,
    split_name: str,
    index: int,
    num_shards: int,
    fingerprint: str,
) -> list[Path] | None:
    """Get paths of shards with different fingerprints for the given index."""
    # Get the path to the shard
    shard_file_name = SHARD_FILE_NAME_TEMPLATE.format(
        split=split_name,
        index=index,
        num_shards=num_shards,
        fingerprint=fingerprint,
    )
    shard_path = output_dir.joinpath(shard_file_name)

    # Get any other shard with the same index
    other_shards_for_same_index = list(
        output_dir.glob(
            SHARD_FILE_NAME_TEMPLATE.format(
                split=split_name,
                index=index,
                num_shards="*",
                fingerprint="*",
            )
        )
    )

    # Remove the desired fingerprinted shard from the list
    with suppress(ValueError):
        other_shards_for_same_index.remove(shard_path)

    # If the list is not empty, then there is more than one shard with the same index
    if other_shards_for_same_index:
        return other_shards_for_same_index
    return None


def _create_parquet_files_for_dataset_split(
    *,
    dataset_split: datasets.Dataset,
    split_name: str,
    num_shards: int,
    embed_external_files: bool = True,
    shards_output_dir: Path,
) -> None:
    """Create parquet files for a given dataset split.

    This saves the parquet files to a given directory. This will delete shards that have the same
    index and a different fingerprint.
    """
    shards = _shard_dataset(
        dataset=dataset_split, num_shards=num_shards, embed_external_files=embed_external_files
    )

    # Create an iterator with a progress bar for the shards
    shard_iterator = tqdm(
        enumerate(shards), desc="Creating parquet files for shards", total=num_shards
    )

    for index, shard in shard_iterator:
        shard_file_name = SHARD_FILE_NAME_TEMPLATE.format(
            split=split_name,
            index=index,
            num_shards=num_shards,
            fingerprint=shard._fingerprint,  # noqa: SLF001
        )
        shard_path = shards_output_dir.joinpath(shard_file_name)

        # Delete shards with a different fingerprint
        _delete_shards_with_different_fingerprint(
            output_dir=shards_output_dir,
            split_name=split_name,
            index=index,
            num_shards=num_shards,
            fingerprint=shard._fingerprint,  # noqa: SLF001
        )

        shard_path.parent.mkdir(parents=True, exist_ok=True)
        shard.to_parquet(shard_path)
        logger.info(f"Created parquet file: `{shard_path}`")


def _export_dataset_as_parquet_files_for_hub(
    *,
    dataset: datasets.DatasetDict,
    dataset_shards_output_dir: Path,
    num_shards: dict[str, int],
    embed_external_files: bool = True,
) -> None:
    """Export a dataset as parquet files for easier uploading to the hub."""
    for split_name, dataset_split in dataset.items():
        logging.info(f"Creating parquet files for split `{split_name}`")
        _create_parquet_files_for_dataset_split(
            dataset_split=dataset_split,
            split_name=split_name,
            num_shards=num_shards[split_name],
            embed_external_files=embed_external_files,
            shards_output_dir=dataset_shards_output_dir,
        )


def _upload_parquet_files_to_hub(
    *,
    hf_repo_id: str,
    config_name: str,
    dataset_shards_output_dir: Path,
    is_private_repo: bool = True,
    use_multi_commits: bool = False,
) -> None:
    """Upload parquet files to the hub."""
    # Ensure that the parquet files directory exists and is the same as the config name
    assert dataset_shards_output_dir.is_dir()
    assert dataset_shards_output_dir.name == config_name

    api = HfApi(endpoint=HF_ENDPOINT)

    token = HfFolder.get_token()
    if token is None:
        raise OSError(
            "You need to provide a `token` or be logged in to Hugging Face with `huggingface-cli"
            " login`."
        )

    api.create_repo(
        hf_repo_id,
        token=token,
        repo_type="dataset",
        private=is_private_repo,
        exist_ok=True,
    )

    logger.info("Starting the upload...")
    api.upload_folder(
        folder_path=dataset_shards_output_dir,
        repo_id=hf_repo_id,
        repo_type="dataset",
        path_in_repo=config_name,
        allow_patterns="*.parquet",
        delete_patterns="*.parquet",
        commit_message=f"Upload {config_name} with huggingface_hub",
        multi_commits=use_multi_commits,
        multi_commits_verbose=use_multi_commits,
    )

    logger.info("Finished uploading the parquet files to the hub.")


def _upload_dataset_to_hub_using_hf(
    *,
    dataset: datasets.DatasetDict,
    config_name: str,
    hf_repo_id: str,
    num_shards: dict[str, int],
) -> None:
    """Upload a dataset to the hub using `push_to_hub`."""
    try:
        dataset.push_to_hub(hf_repo_id, config_name=config_name, num_shards=num_shards)
    except BadRequestError as request_error:
        if request_error.server_message is not None and "YAML" in request_error.server_message:
            logger.error("Invalid YAML occurred while pushing dataset to the hub.")
        else:
            raise request_error from None
    except HfHubHTTPError as http_error:
        if http_error.errno == 429:  # noqa: PLR2004
            logger.error("Rate limited while pushing dataset to the hub.")
            return


def upload_dataset_to_hub(
    saved_hf_dataset_dir: Path,
    *,
    hf_repo_id: str,
    num_shards: dict[str, int],
    use_custom_method: bool = False,
    hf_parquets_dir: Path | None = None,
) -> None:
    """Upload the dataset to the hub."""
    logger.info("Load dataset from disk...")
    dataset_dict = datasets.load_from_disk(str(saved_hf_dataset_dir))
    assert isinstance(dataset_dict, datasets.DatasetDict)

    # Get the config name for the dataset
    config_name = next(iter(dataset_dict.values())).info.config_name

    if use_custom_method:
        logger.info("Using the 'custom method' to push the dataset to the hub.")
        if not hf_parquets_dir:
            raise ValueError(
                "You need to provide the `hf_parquets_dir` when using the custom method."
            )
        dataset_shards_output_dir = hf_parquets_dir.joinpath(config_name)
        _export_dataset_as_parquet_files_for_hub(
            dataset=dataset_dict,
            dataset_shards_output_dir=dataset_shards_output_dir,
            num_shards=num_shards,
        )
        _upload_parquet_files_to_hub(
            hf_repo_id=hf_repo_id,
            config_name=config_name,
            dataset_shards_output_dir=dataset_shards_output_dir,
        )
    else:
        logger.info("Using the 'push_to_hub' method to push the dataset to the hub.")
        _upload_dataset_to_hub_using_hf(
            dataset=dataset_dict,
            config_name=config_name,
            hf_repo_id=hf_repo_id,
            num_shards=num_shards,
        )

    logger.info(f"Finished pushing dataset {config_name} to the hub.")
