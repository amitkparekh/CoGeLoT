from pathlib import Path
from typing import Annotated

import typer

from cogelot.common.hf_datasets import upload_dataset_to_hub
from cogelot.entrypoints.settings import Settings


settings = Settings()

MaxShardSize = Annotated[str, typer.Option(help="Maximum shard size for the dataset")]
HFRepoId = Annotated[str, typer.Option(help="Repository ID for the dataset on HF")]


def upload_raw_dataset(
    parsed_hf_dataset_dir: Annotated[
        Path, typer.Argument(help="Location of the raw HF dataset")
    ] = settings.parsed_hf_dataset_dir,
    max_shard_size: MaxShardSize = settings.max_shard_size,
    hf_repo_id: HFRepoId = settings.hf_repo_id,
) -> None:
    """Upload the raw dataset to the hub."""
    upload_dataset_to_hub(
        parsed_hf_dataset_dir,
        hf_repo_id,
        max_shard_size,
        config_name=settings.raw_config_name,
    )


def upload_preprocessed_dataset(
    preprocessed_hf_dataset_dir: Annotated[
        Path, typer.Argument(help="Location of the preprocessed HF dataset")
    ] = settings.preprocessed_hf_dataset_dir,
    max_shard_size: MaxShardSize = settings.max_shard_size,
    hf_repo_id: HFRepoId = settings.hf_repo_id,
) -> None:
    """Upload the raw dataset to the hub."""
    upload_dataset_to_hub(
        preprocessed_hf_dataset_dir,
        hf_repo_id,
        max_shard_size,
        config_name=settings.preprocessed_config_name,
    )
