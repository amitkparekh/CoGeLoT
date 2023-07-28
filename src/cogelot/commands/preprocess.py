from collections.abc import Iterator
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hydra
import typer
from datasets import DatasetDict, load_from_disk
from loguru import logger
from omegaconf import DictConfig
from pydantic import BaseSettings
from rich import progress
from torch.utils.data import DataLoader, Dataset
from tqdm.rich import RateColumn

from cogelot.common.io import save_pickle
from cogelot.data.datasets import (
    create_hf_dataset,
    create_validation_split,
    generate_preprocess_instances_for_hf_dataset,
    set_dataset_format,
)
from cogelot.data.parse import get_all_raw_instance_directories, parse_and_save_instance
from cogelot.structures.vima import VIMAInstance


if TYPE_CHECKING:
    from cogelot.data.preprocess import InstancePreprocessor


class PreprocessSettings(BaseSettings):
    """Settings for the preprocessing command."""

    total_num_instances: int = 660103

    # Location of the configs relative to project root
    config_dir: Path = Path("configs/")

    # Location of the storage data relative to the project root
    storage_data_dir: Path = Path("storage/data/")
    raw_data_dir: Path = storage_data_dir.joinpath("raw/vima_v6/")
    normalized_data_dir: Path = storage_data_dir.joinpath("normalized/")
    preprocessed_data_dir: Path = storage_data_dir.joinpath("preprocessed/")
    hf_dataset_dir: Path = storage_data_dir.joinpath("hf/")

    seed: int = 1000

    # Maximum num. of validation instances across all the tasks.
    max_num_validation_instances: int = 50000

    instance_preprocessor_hydra_config: Path = config_dir.joinpath("instance_preprocessor.yaml")


settings = PreprocessSettings()

app = typer.Typer(add_completion=False)


def create_progress() -> progress.Progress:
    """Create a progress bar."""
    progress_bar = progress.Progress(
        "[progress.description]{task.description}",
        progress.BarColumn(bar_width=None),
        progress.MofNCompleteColumn(),
        cast(progress.ProgressColumn, RateColumn(unit="it")),
        progress.TimeElapsedColumn(),
        progress.TimeRemainingColumn(),
    )

    return progress_bar


class InstancePreprocessDataset(Dataset[None]):
    """Preprocess instance using multiprocessing."""

    def __init__(
        self, normalized_instance_paths: list[Path], output_dir: Path, config: DictConfig
    ) -> None:
        super().__init__()

        self._output_dir = output_dir

        self._normalized_instance_paths = normalized_instance_paths
        self._instance_preprocessor: InstancePreprocessor = hydra.utils.instantiate(
            config["instance_preprocessor"]
        )

    def __len__(self) -> int:
        """Total number of instnaces."""
        return len(self._normalized_instance_paths)

    def __getitem__(self, index: int) -> None:
        """Preprocess the instance and save it."""
        instance_path = self._normalized_instance_paths[index]

        if not instance_path.stat().st_size:
            logger.error(f"Skipping empty instance: {instance_path}")
            return

        # Preprocess the instance
        instance = VIMAInstance.load(instance_path)
        preprocessed_instance = self._instance_preprocessor.preprocess(instance)

        # Create the path for the preprocessed instance
        preprocessed_instance_path = self._output_dir.joinpath(
            f"{instance.task}/{instance.task}_{index}.pkl"
        )
        preprocessed_instance_path.parent.mkdir(parents=True, exist_ok=True)
        save_pickle(preprocessed_instance, preprocessed_instance_path, compress=True)


def get_raw_instance_directories(
    raw_data_root: Path, *, progress_bar: progress.Progress, task_id: progress.TaskID
) -> Iterator[Path]:
    """Yield all the raw instance directories."""
    path_iterator = get_all_raw_instance_directories(raw_data_root)

    for path in path_iterator:
        yield path
        progress_bar.advance(task_id)


def get_instance_paths(
    data_root: Path,
    *,
    instance_suffix: str,
    progress_bar: progress.Progress,
    task_id: progress.TaskID,
) -> Iterator[Path]:
    """Yield instances from root."""
    path_iterator = data_root.glob(f"*/*.{instance_suffix}*")

    for path in path_iterator:
        yield path
        progress_bar.advance(task_id)


@app.command(name="normalize")
def normalize_raw_data(
    raw_data_root: Path = typer.Argument(
        settings.raw_data_dir, help="Root directory of the raw data.", envvar="RAW_DATA_ROOT"
    ),
    output_dir: Path = typer.Argument(
        settings.normalized_data_dir, help="Output directory.", envvar="NORMALIZED_DATA_DIR"
    ),
    num_workers: int = typer.Option(1, help="Number of workers."),
    multiprocessing_chunksize: int = typer.Option(1, help="Chunksize for imap."),
    *,
    delete_raw_instances: bool = typer.Option(
        default=False, help="Whether to delete the raw instances after normalization."
    ),
    replace_if_exists: bool = typer.Option(
        default=False, help="Replace the normalized file if exists."
    ),
) -> None:
    """Normalize the raw data."""
    progress_bar = create_progress()

    with progress_bar:
        get_raw_instance_dir_task = progress_bar.add_task(
            "Collect raw instance directories", total=None
        )
        normalize_instance_task = progress_bar.add_task("Normalize instances", total=None)

        raw_instance_directory_iterator = list(
            get_raw_instance_directories(
                raw_data_root, progress_bar=progress_bar, task_id=get_raw_instance_dir_task
            )
        )
        progress_bar.update(get_raw_instance_dir_task, total=len(raw_instance_directory_iterator))
        progress_bar.update(normalize_instance_task, total=len(raw_instance_directory_iterator))

        parse_and_save_partial = partial(
            parse_and_save_instance,
            output_dir=output_dir,
            delete_raw_instance_dir=delete_raw_instances,
            replace_if_exists=replace_if_exists,
        )

        with Pool(num_workers) as pool:
            iterator = pool.imap_unordered(
                parse_and_save_partial,
                raw_instance_directory_iterator,
                chunksize=multiprocessing_chunksize,
            )

            for _ in iterator:
                progress_bar.advance(normalize_instance_task)


@app.command(name="preprocess")
def preprocess_normalized_data(
    normalized_data_root: Path = typer.Argument(
        settings.normalized_data_dir,
        help="Root directory for the normalized data",
        envvar="NORMALIZED_DATA_ROOT",
    ),
    output_dir: Path = typer.Argument(
        settings.preprocessed_data_dir, help="Output directory.", envvar="PREPROCESSED_DATA_DIR"
    ),
    num_workers: int = typer.Option(1, help="Number of workers."),
    instance_preprocessor_hydra_config: Path = typer.Option(
        settings.instance_preprocessor_hydra_config,
        help="Hydra config for the instance preprocessor.",
        envvar="INSTANCE_PREPROCESSOR_HYDRA_CONFIG",
    ),
) -> None:
    """Preprocess all the instances to be ready for modelling."""
    progress_bar = create_progress()

    logger.info("Load instance preprocessor config from file")
    with hydra.initialize_config_dir(
        config_dir=str(instance_preprocessor_hydra_config.parent.resolve()), version_base="1.3"
    ):
        config = hydra.compose(config_name=instance_preprocessor_hydra_config.name)

    with progress_bar:
        get_normalized_instance_paths_task = progress_bar.add_task(
            "Collect normalized instance paths", total=None
        )
        preprocess_instance_task = progress_bar.add_task("Preprocess instances", total=None)

        normalized_instance_paths = list(
            get_instance_paths(
                normalized_data_root,
                instance_suffix="json",
                progress_bar=progress_bar,
                task_id=get_normalized_instance_paths_task,
            )
        )
        progress_bar.update(
            get_normalized_instance_paths_task, total=len(normalized_instance_paths)
        )
        progress_bar.update(preprocess_instance_task, total=len(normalized_instance_paths))

        dataset = InstancePreprocessDataset(normalized_instance_paths, output_dir, config)
        dataloader = DataLoader(
            dataset=dataset, num_workers=num_workers, batch_size=None, batch_sampler=None
        )
        for _ in dataloader:
            progress_bar.advance(preprocess_instance_task)


@app.command(name="convert-to-hf")
def convert_to_hf_dataset(
    preprocessed_instances_root: Path = typer.Argument(
        settings.preprocessed_data_dir,
        help="Root directory for the preprocessed data",
        envvar="PREPROCESSED_DATA_DIR",
    ),
    hf_dataset_dir: Path = typer.Argument(
        settings.hf_dataset_dir, help="Output directory.", envvar="HF_DATASET_DIR"
    ),
    max_num_validation_instances: int = typer.Option(
        default=settings.max_num_validation_instances,
        help="Maximum number of validation instances.",
        envvar="MAX_NUM_VALIDATION_INSTANCES",
    ),
    num_workers: int = typer.Option(1, help="Number of workers."),
    seed: int = typer.Option(settings.seed, help="Seed for the random number generator."),
    writer_batch_size: int = typer.Option(
        default=1000, help="Writer batch size when creating the split."
    ),
    max_shard_size: str = typer.Option("2GB", help="Maximum shard size for the dataset."),
) -> None:
    """Create a HuggingFace dataset from the preprocessed instances."""
    hf_dataset = create_hf_dataset(
        generate_preprocess_instances_for_hf_dataset,
        num_workers=num_workers,
        preprocessed_instances=list(preprocessed_instances_root.glob("*/*.pkl*")),
    )

    hf_dataset = set_dataset_format(hf_dataset)
    split_dataset = create_validation_split(
        hf_dataset,
        max_num_validation_instances=max_num_validation_instances,
        seed=seed,
        writer_batch_size=writer_batch_size,
    )
    split_dataset.save_to_disk(hf_dataset_dir, num_proc=num_workers, max_shard_size=max_shard_size)


@app.command(name="upload-to-hub")
def upload_to_hub(
    hf_dataset_dir: Path = typer.Argument(
        settings.hf_dataset_dir, help="Output directory.", envvar="HF_DATASET_DIR"
    ),
    repo_id: str = typer.Option("amitkparekh/vima", help="Repository ID."),
    max_shard_size: str = typer.Option("2GB", help="Maximum shard size for the dataset."),
    debug_dataset_size: int = typer.Option(
        default=100, help="Num. instances in the debug dataset."
    ),
) -> None:
    """Upload the dataset to the HuggingFace Hub."""
    dataset = load_from_disk(str(hf_dataset_dir.resolve()))
    dataset.push_to_hub(repo_id, max_shard_size=max_shard_size)

    # Also upload a small version of the dataset for debugging
    assert isinstance(dataset, DatasetDict)
    small_dataset = DatasetDict(
        {split: dataset[split].select(range(debug_dataset_size)) for split in dataset}
    )
    small_dataset.push_to_hub(f"{repo_id}-small", max_shard_size=max_shard_size)


if __name__ == "__main__":
    app()
