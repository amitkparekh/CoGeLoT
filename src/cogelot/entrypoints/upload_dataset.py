from pathlib import Path
from typing import Annotated, Optional

import typer
from loguru import logger

from cogelot.common.hf_datasets import upload_dataset_to_hub
from cogelot.entrypoints.settings import Settings
from cogelot.structures.vima import Task


settings = Settings()


def upload_raw_dataset(
    parsed_hf_dataset_dir: Annotated[
        Path, typer.Argument(help="Location of the raw HF dataset")
    ] = settings.parsed_hf_dataset_dir,
    parsed_hf_parquets_dir: Annotated[
        Path, typer.Argument(help="Location of the raw HF parquets")
    ] = settings.parsed_hf_parquets_dir,
    task_index_filter: Annotated[
        Optional[int], typer.Option(min=Task.minimum(), max=Task.maximum())  # noqa: UP007
    ] = None,
    *,
    use_custom_method: Annotated[
        bool,
        typer.Option(help="Use the custom method which is likely faster, but might be unstable."),
    ] = False,
) -> None:
    """Upload the raw datasets to the hub."""
    logger.info("Uploading the raw datasets to the hub...")

    parsed_hf_parquets_dir.mkdir(parents=True, exist_ok=True)

    for task_dataset_path in parsed_hf_dataset_dir.iterdir():
        task = Task[task_dataset_path.name]
        if task_index_filter is not None and task_index_filter != task.value:
            logger.info(f"Skipping task {task}...")
            continue

        logger.info(f"Uploading raw instances from the the {task} dataset...")

        upload_dataset_to_hub(
            task_dataset_path,
            hf_repo_id=settings.hf_repo_id,
            num_shards=settings.num_shards,
            parquet_files_dir_for_dataset=parsed_hf_parquets_dir.joinpath(task.name),
            use_custom_method=use_custom_method,
        )

    logger.info("Finished uploading the raw datasets to the hub.")


def upload_preprocessed_dataset(
    preprocessed_hf_dataset_dir: Annotated[
        Path, typer.Argument(help="Location of the preprocessed HF dataset")
    ] = settings.preprocessed_hf_dataset_dir,
    preprocessed_hf_parquets_dir: Annotated[
        Path, typer.Argument(help="Location of the preprocessed HF parquets")
    ] = settings.preprocessed_hf_parquets_dir,
    task_index_filter: Annotated[
        Optional[int], typer.Option(min=Task.minimum(), max=Task.maximum())  # noqa: UP007
    ] = None,
    *,
    use_custom_method: Annotated[
        bool,
        typer.Option(help="Use the custom method which is likely faster, but might be unstable."),
    ] = False,
) -> None:
    """Upload the preprocessed datasets to the hub."""
    logger.info("Uploading the preprocessed datasets to the hub...")

    preprocessed_hf_parquets_dir.mkdir(parents=True, exist_ok=True)

    for task_dataset_path in preprocessed_hf_dataset_dir.iterdir():
        task = Task[task_dataset_path.name.removeprefix("preprocessed--")]
        if task_index_filter is not None and task_index_filter != task.value:
            logger.info(f"Skipping task {task}...")
            continue

        logger.info(f"Uploading preprocessed instances from the the {task} dataset...")
        upload_dataset_to_hub(
            task_dataset_path,
            hf_repo_id=settings.hf_repo_id,
            num_shards=settings.num_shards,
            parquet_files_dir_for_dataset=preprocessed_hf_parquets_dir.joinpath(task.name),
            use_custom_method=use_custom_method,
        )

    logger.info("Finished uploading the preprocessed datasets to the hub.")
