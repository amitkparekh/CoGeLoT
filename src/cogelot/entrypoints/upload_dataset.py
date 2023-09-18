from collections.abc import Callable
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import typer

from cogelot.common.hf_datasets import upload_dataset_to_hub
from cogelot.entrypoints.settings import Settings


settings = Settings()


def _create_partial_fn_for_upload(
    hf_repo_id: str = settings.hf_repo_id, num_shards: dict[str, int] = settings.num_shards
) -> Callable[[Path], None]:
    """Create the partial function for uploading a dataset to the hub."""
    return partial(
        upload_dataset_to_hub,
        hf_repo_id=hf_repo_id,
        num_shards=num_shards,
    )


def upload_raw_dataset(
    parsed_hf_dataset_dir: Annotated[
        Path, typer.Argument(help="Location of the raw HF dataset")
    ] = settings.parsed_hf_dataset_dir,
    num_simultaneous_uploads: Annotated[
        int, typer.Option(help="Number of tasks to upload simultaneously (via multiprocessing)")
    ] = 1,
) -> None:
    """Upload the raw dataset to the hub."""
    partial_upload_fn = _create_partial_fn_for_upload()
    with Pool(num_simultaneous_uploads) as pool:
        pool.map(partial_upload_fn, parsed_hf_dataset_dir.iterdir())


def upload_preprocessed_dataset(
    preprocessed_hf_dataset_dir: Annotated[
        Path, typer.Argument(help="Location of the preprocessed HF dataset")
    ] = settings.preprocessed_hf_dataset_dir,
    num_simultaneous_uploads: Annotated[
        int, typer.Option(help="Number of tasks to upload simultaneously (via multiprocessing)")
    ] = 1,
) -> None:
    """Upload the raw dataset to the hub."""
    partial_upload_fn = _create_partial_fn_for_upload()
    with Pool(num_simultaneous_uploads) as pool:
        pool.map(partial_upload_fn, preprocessed_hf_dataset_dir.iterdir())
