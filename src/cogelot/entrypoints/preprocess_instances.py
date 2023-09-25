from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import datasets
import hydra
import typer
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cogelot.common.io import save_pickle
from cogelot.entrypoints.settings import Settings
from cogelot.modules.instance_preprocessor import InstancePreprocessor
from cogelot.structures.vima import Task, VIMAInstance


settings = Settings()


def _instantiate_instance_preprocessor_from_config(config_file: Path) -> InstancePreprocessor:
    """Instnatiate the instance preprocessor from the config file."""
    with hydra.initialize_config_dir(
        config_dir=str(config_file.parent.resolve()), version_base="1.3"
    ):
        config = hydra.compose(config_name=config_file.name)
        instance_preprocessor: InstancePreprocessor = hydra.utils.instantiate(
            config["instance_preprocessor"]
        )
    return instance_preprocessor


def _load_parsed_datasets_for_each_task(
    parsed_hf_datasets_dir: Path,
) -> Iterator[tuple[Task, datasets.DatasetDict]]:
    """Load the parsed datasets for each task."""
    dataset_dir_per_task = list(parsed_hf_datasets_dir.iterdir())
    logger.info(f"Loading {len(dataset_dir_per_task)} datasets from {parsed_hf_datasets_dir}...")

    for task_dataset_dir in dataset_dir_per_task:
        task = Task[task_dataset_dir.name]

        logger.info(f"Loading dataset for {task} from {task_dataset_dir}...")
        dataset = datasets.load_from_disk(str(task_dataset_dir))

        logger.info(f"Loaded dataset for {task}.")
        assert isinstance(dataset, datasets.DatasetDict)
        yield task, dataset


class InstancePreprocessorDataset(Dataset[None]):
    """Preprocess VIMA instances for the model."""

    def __init__(
        self,
        *,
        raw_dataset: datasets.Dataset,
        instance_preprocessor: InstancePreprocessor,
        output_dir: Path,
        replace_if_exists: bool = False,
    ) -> None:
        self.raw_dataset = raw_dataset
        self.instance_preprocessor = instance_preprocessor

        self._output_dir = output_dir
        self._replace_if_exists = replace_if_exists

    def __len__(self) -> int:
        """Total number of instances."""
        return len(self.raw_dataset)

    def __getitem__(self, index: int) -> None:
        """Preprocess and return the instance."""
        raw_instance = self.raw_dataset[index]
        # Create the path for the preprocessed instance
        task = Task(raw_instance["task"])
        instance_index = raw_instance["index"]
        preprocessed_instance_path = self._output_dir.joinpath(f"{task.name}_{instance_index}.pkl")

        # If the path exists, we don't need to preprocess it
        if preprocessed_instance_path.exists() and not self._replace_if_exists:
            return

        vima_instance = VIMAInstance.model_validate(raw_instance)

        # Preprocess the instance
        preprocessed_instance = self.instance_preprocessor.preprocess(vima_instance)

        # Save the preprocessed instance
        preprocessed_instance_path.parent.mkdir(parents=True, exist_ok=True)
        save_pickle(preprocessed_instance.model_dump(), preprocessed_instance_path, compress=True)


def create_dataloader_to_preprocess_instances(
    *,
    dataset_split: datasets.Dataset,
    instance_preprocessor: InstancePreprocessor,
    num_workers: int,
    output_dir: Path,
) -> DataLoader[None]:
    """Create a dataloader to preprocess the instances as fast as possible."""
    dataloader = DataLoader[None](
        dataset=InstancePreprocessorDataset(
            raw_dataset=dataset_split,
            instance_preprocessor=instance_preprocessor,
            output_dir=output_dir,
        ),
        batch_size=None,
        batch_sampler=None,
        num_workers=num_workers,
    )
    return dataloader


def preprocess_instances_for_task(
    *,
    task: Task,
    dataset: datasets.DatasetDict,
    instance_preprocessor: InstancePreprocessor,
    preprocessed_instances_output_dir: Path,
    num_workers: int,
) -> None:
    """Create the preprocessed instances for the given task (and dataset).

    Convert the parsed instances into the preprocessed ones, and pickle each preprocessed instance
    to disk.
    """
    preprocessed_instances_output_dir = preprocessed_instances_output_dir.joinpath(task.name)

    logger.info(f"Creating dataloaders for each {task} split...")
    train_dataloader = create_dataloader_to_preprocess_instances(
        dataset_split=dataset["train"],
        instance_preprocessor=instance_preprocessor,
        num_workers=num_workers,
        output_dir=preprocessed_instances_output_dir.joinpath("train/"),
    )
    valid_dataloader = create_dataloader_to_preprocess_instances(
        dataset_split=dataset["valid"],
        instance_preprocessor=instance_preprocessor,
        num_workers=num_workers,
        output_dir=preprocessed_instances_output_dir.joinpath("valid/"),
    )

    logger.info(f"Creating iterators to process each {task} split...")
    train_iterator = tqdm(
        train_dataloader, total=len(dataset["train"]), desc="Preprocess train instances"
    )
    valid_iterator = tqdm(
        valid_dataloader, total=len(dataset["valid"]), desc="Preprocess valid instances"
    )

    logger.info(f"Preprocessing the {task} instances...")
    list(train_iterator)
    list(valid_iterator)

    logger.info(f"Finished preprocessing the {task} instances.")


def preprocess_instances(
    parsed_hf_dataset_dir: Annotated[
        Path, typer.Argument(help="Where to get the parsed HF datasets (for each task)")
    ] = settings.parsed_hf_dataset_dir,
    preprocessed_instances_dir: Annotated[
        Path, typer.Option(help="Directory to save the preprocessed instances to.")
    ] = settings.preprocessed_instances_dir,
    instance_preprocessor_hydra_config: Annotated[
        Path, typer.Option(help="Hydra config for the instance preprocessor.")
    ] = settings.instance_preprocessor_hydra_config,
    num_workers: Annotated[int, typer.Option(help="Number of workers.")] = 1,
) -> None:
    """Preprocess all the parsed instances per task."""
    logger.info("Loading the instance preprocessor...")
    instance_preprocessor = _instantiate_instance_preprocessor_from_config(
        instance_preprocessor_hydra_config
    )

    logger.info("Loading the parsed datasets for each task...")
    parsed_datasets_per_task_iterator = _load_parsed_datasets_for_each_task(parsed_hf_dataset_dir)

    for task, dataset in parsed_datasets_per_task_iterator:
        logger.info(f"Preprocessing the {task} instances...")
        preprocess_instances_for_task(
            task=task,
            dataset=dataset,
            instance_preprocessor=instance_preprocessor,
            preprocessed_instances_output_dir=preprocessed_instances_dir,
            num_workers=num_workers,
        )
        logger.info(f"Finished preprocessing the {task} instances.")
