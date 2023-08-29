from functools import partial
from pathlib import Path
from typing import Annotated

import datasets
import hydra
import typer
from loguru import logger
from rich.progress import track
from torch.utils.data import DataLoader, Dataset

from cogelot.common.hf_datasets import (
    download_parquet_files_from_hub,
    get_location_of_parquet_files,
    load_dataset_from_parquet_files,
)
from cogelot.common.io import load_pickle, save_pickle
from cogelot.common.rich import create_progress_bar
from cogelot.data.datasets import create_hf_dataset_from_paths, load_instance_from_pickled_path
from cogelot.entrypoints.settings import Settings
from cogelot.modules.instance_preprocessor import InstancePreprocessor
from cogelot.structures.model import PreprocessedInstance
from cogelot.structures.vima import VIMAInstance


settings = Settings()


def get_all_preprocessed_instance_paths(preprocessed_instance_dir: Path) -> list[Path]:
    """Get all the parsed VIMA instance paths."""
    path_iterator = preprocessed_instance_dir.rglob("*.pkl.gz")
    return list(track(path_iterator, description="Getting all preprocessed instance paths"))


load_preprocessed_instance_from_path_fn = partial(
    load_instance_from_pickled_path,
    instance=PreprocessedInstance,
    load_from_path_fn=load_pickle,
)


class InstancePreprocessorDataset(Dataset[None]):
    """Preprocess VIMA instances for the model."""

    def __init__(
        self,
        *,
        raw_dataset: datasets.Dataset,
        instance_preprocessor: InstancePreprocessor,
        output_dir: Path,
    ) -> None:
        self.raw_dataset = raw_dataset
        self.instance_preprocessor = instance_preprocessor

        self._output_dir = output_dir

    def __len__(self) -> int:
        """Total number of instances."""
        return len(self.raw_dataset)

    def __getitem__(self, index: int) -> None:
        """Preprocess and return the instance."""
        vima_instance = VIMAInstance.model_validate(self.raw_dataset[index])

        # Preprocess the instance
        preprocessed_instance = self.instance_preprocessor.preprocess(vima_instance)

        # Create the path for the preprocessed instance
        preprocessed_instance_path = self._output_dir.joinpath(
            f"{vima_instance.task}/{vima_instance.task}_{index}.pkl"
        )
        preprocessed_instance_path.parent.mkdir(parents=True, exist_ok=True)
        save_pickle(preprocessed_instance, preprocessed_instance_path, compress=True)


def download_and_load_raw_dataset(
    repo_id: str, max_workers: int, config_name: str
) -> datasets.DatasetDict:
    """Download and load the raw dataset."""
    download_parquet_files_from_hub(repo_id, max_workers=max_workers)
    parquet_files_location = get_location_of_parquet_files(repo_id)
    return load_dataset_from_parquet_files(
        parquet_files_location, num_proc=max_workers, name=config_name
    )


def instantiate_instance_preprocessor_from_config(config_file: Path) -> InstancePreprocessor:
    """Instnatiate the instance preprocessor from the config file."""
    with hydra.initialize_config_dir(
        config_dir=str(config_file.parent.resolve()), version_base="1.3"
    ):
        config = hydra.compose(config_name=config_file.name)
        instance_preprocessor: InstancePreprocessor = hydra.utils.instantiate(
            config["instance_preprocessor"]
        )
    return instance_preprocessor


def create_dataloader_to_preprocess_instances(
    *,
    raw_dataset_split: datasets.Dataset,
    instance_preprocessor: InstancePreprocessor,
    num_workers: int,
    output_dir: Path,
) -> DataLoader[None]:
    """Create a dataloader to preprocess the instances as fast as possible."""
    dataloader = DataLoader[None](
        dataset=InstancePreprocessorDataset(
            raw_dataset=raw_dataset_split,
            instance_preprocessor=instance_preprocessor,
            output_dir=output_dir,
        ),
        batch_size=None,
        batch_sampler=None,
        num_workers=num_workers,
    )
    return dataloader


def create_and_cache_preprocessed_instances(
    instance_preprocessor_hydra_config: Path,
    preprocessed_instances_dir: Path,
    num_workers: int,
    hf_repo_id: str,
) -> None:
    """Create the preprocessed instances from the raw instances, and save them to disk."""
    logger.info("Downloading and loading the raw dataset...")
    raw_dataset = download_and_load_raw_dataset(
        hf_repo_id, max_workers=num_workers, config_name=settings.raw_config_name
    )

    logger.info("Loading the instance preprocessor...")
    instance_preprocessor = instantiate_instance_preprocessor_from_config(
        instance_preprocessor_hydra_config
    )

    logger.info("Creating dataloaders for each split...")
    train_dataloader = create_dataloader_to_preprocess_instances(
        raw_dataset_split=raw_dataset["train"],
        instance_preprocessor=instance_preprocessor,
        num_workers=num_workers,
        output_dir=preprocessed_instances_dir.joinpath("train/"),
    )
    valid_dataloader = create_dataloader_to_preprocess_instances(
        raw_dataset_split=raw_dataset["valid"],
        instance_preprocessor=instance_preprocessor,
        num_workers=num_workers,
        output_dir=preprocessed_instances_dir.joinpath("valid/"),
    )

    logger.info("Creating iterators to process each split...")
    progress_bar = create_progress_bar()
    train_iterator = progress_bar.track(
        train_dataloader, total=len(raw_dataset["train"]), description="Preprocess train instances"
    )
    valid_iterator = progress_bar.track(
        valid_dataloader, total=len(raw_dataset["valid"]), description="Preprocess valid instances"
    )

    logger.info("Preprocessing the instances...")
    with progress_bar:
        list(train_iterator)
        list(valid_iterator)

    logger.info("Finished preprocessing the instances.")


def create_preprocessed_hf_dataset(
    preprocessed_instances_dir: Path,
    hf_repo_id: str,
    num_workers: int,
    max_shard_size: str,
) -> None:
    """Preprocess the instances and upload the preprocesed dataset to HF."""
    logger.info("Getting all the preprocessed instance paths...")
    all_train_instance_paths = get_all_preprocessed_instance_paths(
        preprocessed_instances_dir.joinpath("train/")
    )
    all_valid_instance_paths = get_all_preprocessed_instance_paths(
        preprocessed_instances_dir.joinpath("valid/")
    )

    logger.info("Creating the HF dataset for each split...")
    preprocessed_train_dataset = create_hf_dataset_from_paths(
        paths=all_train_instance_paths,
        load_instance_from_path_fn=load_preprocessed_instance_from_path_fn,
        dataset_features=PreprocessedInstance.dataset_features(),
        num_workers=num_workers,
        writer_batch_size=settings.writer_batch_size,
    )
    preprocessed_valid_dataset = create_hf_dataset_from_paths(
        paths=all_valid_instance_paths,
        load_instance_from_path_fn=load_preprocessed_instance_from_path_fn,
        dataset_features=PreprocessedInstance.dataset_features(),
        num_workers=num_workers,
        writer_batch_size=settings.writer_batch_size,
    )
    # Merge the two into a dataset dict
    dataset_dict = datasets.DatasetDict(
        {"train": preprocessed_train_dataset, "valid": preprocessed_valid_dataset}
    )

    logger.info("Pushing the preprocessed dataset to the hub...")
    dataset_dict.push_to_hub(
        hf_repo_id, max_shard_size=max_shard_size, config_name=settings.preprocessed_config_name
    )


def create_preprocessed_dataset(
    instance_preprocessor_hydra_config: Annotated[
        Path, typer.Option(help="Hydra config for the instance preprocessor.")
    ] = settings.instance_preprocessor_hydra_config,
    preprocessed_instances_dir: Annotated[
        Path, typer.Option(help="Directory to save the preprocessed instances to.")
    ] = settings.preprocessed_data_dir,
    num_workers: Annotated[int, typer.Option(help="Number of workers.")] = 1,
    hf_repo_id: Annotated[
        str, typer.Option(help="Repository ID for the dataset on HF")
    ] = settings.hf_repo_id,
    max_shard_size: Annotated[
        str, typer.Option(help="Maximum shard size for the dataset")
    ] = settings.max_shard_size,
) -> None:
    """Create the preprocessed instances from the raw instances, and upload them to HF.

    Similar to creating the raw instances, we are doing this in steps because things can crash and
    we don't want to keep wasting time debugging an OOM error.
    """
    create_and_cache_preprocessed_instances(
        instance_preprocessor_hydra_config=instance_preprocessor_hydra_config,
        preprocessed_instances_dir=preprocessed_instances_dir,
        num_workers=num_workers,
        hf_repo_id=hf_repo_id,
    )
    create_preprocessed_hf_dataset(
        preprocessed_instances_dir=preprocessed_instances_dir,
        hf_repo_id=hf_repo_id,
        num_workers=num_workers,
        max_shard_size=max_shard_size,
    )


if __name__ == "__main__":
    create_preprocessed_dataset()
