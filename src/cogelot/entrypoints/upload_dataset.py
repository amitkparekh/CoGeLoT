from pathlib import Path
from typing import Annotated

import typer

from cogelot.common.hf_datasets import upload_dataset_to_hub
from cogelot.entrypoints.settings import Settings


settings = Settings()


def upload_raw_dataset(
    parsed_hf_dataset_dir: Annotated[
        Path, typer.Argument(help="Location of the raw HF dataset")
    ] = settings.parsed_hf_dataset_dir,
) -> None:
    """Upload the raw dataset to the hub."""
    for task_dir in parsed_hf_dataset_dir.iterdir():
        upload_dataset_to_hub(
            task_dir,
            hf_repo_id=settings.hf_repo_id,
            num_shards=settings.num_shards,
        )


def upload_preprocessed_dataset(
    preprocessed_hf_dataset_dir: Annotated[
        Path, typer.Argument(help="Location of the preprocessed HF dataset")
    ] = settings.preprocessed_hf_dataset_dir,
) -> None:
    """Upload the raw dataset to the hub."""
    for task_dir in preprocessed_hf_dataset_dir.iterdir():
        upload_dataset_to_hub(
            task_dir,
            hf_repo_id=settings.hf_repo_id,
            num_shards=settings.num_shards,
        )
