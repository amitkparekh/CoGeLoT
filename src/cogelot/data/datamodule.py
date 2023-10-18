import abc
import itertools
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict, Unpack

import datasets
from loguru import logger
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from cogelot.common.hf_datasets import (
    download_parquet_files_from_hub,
    get_location_of_hub_parquet_files,
    load_dataset_from_disk,
    maybe_split_dataset_by_node,
)
from cogelot.data.collate import collate_preprocessed_instances_from_hf_dataset
from cogelot.data.evaluation import VIMAEvaluationDataset
from cogelot.structures.model import EvaluationEpisode, PreprocessedInstance


SetupStage = Literal["fit", "validate", "test", "predict"]


class DataModuleKwargs(TypedDict):
    """Kwargs for the __init__ in the datamodule."""

    num_workers: int
    batch_size: int
    dataloader_kwargs: NotRequired[dict[str, Any]]

    # For filtering the dataset
    task_index_seen: NotRequired[int]
    max_num_instances_seen: NotRequired[int]


class VIMADataModule(abc.ABC, LightningDataModule):
    """Datamodule for the VIMA dataset."""

    def __init__(self, **kwargs: Unpack[DataModuleKwargs]) -> None:
        super().__init__()
        self._kwargs = kwargs
        self._num_workers = kwargs.get("num_workers")
        self.batch_size = kwargs.get("batch_size")
        self._dataloader_kwargs = kwargs.get("dataloader_kwargs", {})

        self.train_dataset: datasets.Dataset
        self.valid_dataset: datasets.Dataset
        self.test_dataset: VIMAEvaluationDataset

    def setup(self, stage: SetupStage) -> None:
        """Setup each GPU to run the data."""
        if stage == "fit":
            dataset = self._load_dataset()
            self.train_dataset = maybe_split_dataset_by_node(dataset["train"])
            self.valid_dataset = maybe_split_dataset_by_node(dataset["valid"])

        if stage == "validate":
            dataset = self._load_dataset()
            self.valid_dataset = maybe_split_dataset_by_node(dataset["valid"])

        if stage == "test":
            self.test_dataset = VIMAEvaluationDataset.from_partition_to_specs()

        self._maybe_filter_datasets()

    def train_dataloader(self) -> DataLoader[list[PreprocessedInstance]]:
        """Create the dataloader for the training set."""
        return DataLoader[list[PreprocessedInstance]](
            self.train_dataset,  # pyright: ignore[reportGeneralTypeIssues]
            batch_size=self.batch_size,
            num_workers=self._num_workers,
            shuffle=True,
            collate_fn=collate_preprocessed_instances_from_hf_dataset,
            **self._dataloader_kwargs,
        )

    def val_dataloader(self) -> DataLoader[list[PreprocessedInstance]]:
        """Create the dataloader for the validation set."""
        return DataLoader[list[PreprocessedInstance]](
            self.valid_dataset,  # pyright: ignore[reportGeneralTypeIssues]
            batch_size=self.batch_size,
            num_workers=self._num_workers,
            shuffle=False,
            collate_fn=collate_preprocessed_instances_from_hf_dataset,
            **self._dataloader_kwargs,
        )

    def test_dataloader(self) -> DataLoader[EvaluationEpisode]:
        """Get a dataloader to use for creating environment during evaluation.

        Disable the `batch_size` and `batch_sampler` to ensure that we get a single instance at a time.
        """
        return DataLoader[EvaluationEpisode](
            self.test_dataset,
            batch_size=None,
            shuffle=False,
            batch_sampler=None,
        )

    @abc.abstractmethod
    def _load_dataset(self) -> datasets.DatasetDict:
        """Load the dataset for all the splits."""
        raise NotImplementedError

    def _maybe_filter_datasets(self) -> None:
        """Apply filtering and such if desired."""
        task_index_seen = self._kwargs.get("task_index_seen")
        max_num_instances_seen = self._kwargs.get("max_num_instances_seen")

        if task_index_seen is not None:
            logger.info(f"Limiting task to `{task_index_seen}`")
            self.train_dataset = self.train_dataset.filter(
                lambda instance: instance["task"] == task_index_seen
            )
            self.valid_dataset = self.valid_dataset.filter(
                lambda instance: instance["task"] == task_index_seen
            )

        if max_num_instances_seen is not None:
            logger.info(f"Selecting `{max_num_instances_seen}` instances")
            selected_indices = itertools.chain(
                *itertools.tee(range(max_num_instances_seen), self.batch_size)
            )
            self.train_dataset = self.train_dataset.select(selected_indices)
            self.valid_dataset = self.valid_dataset.select(selected_indices)


class VIMADataModuleFromHF(VIMADataModule):
    """VIMA DataModule by explicitly downloading the dataset from HF."""

    def __init__(self, *, hf_datasets_repo_name: str, **kwargs: Unpack[DataModuleKwargs]) -> None:
        super().__init__(**kwargs)
        self._hf_datasets_repo_name = hf_datasets_repo_name

    def prepare_data(self) -> None:
        """Prepare any data before starting training.

        This is just making sure the dataset has already been downloaded.
        """
        download_parquet_files_from_hub(self._hf_datasets_repo_name, max_workers=self._num_workers)

    def _load_dataset(self) -> datasets.DatasetDict:
        """Load the dataset from the parquet files.

        This is not using `datasets.load_dataset` because doing it this separate way is much
        faster, but does have some complexity overhead.
        """
        dataset_data_dir = get_location_of_hub_parquet_files(self._hf_datasets_repo_name)
        dataset = load_dataset_from_disk(
            dataset_data_dir,
            extension="parquet",
            config_name="preprocessed",
            num_proc=self._num_workers,
        )
        dataset = dataset.with_format(
            "torch", columns=PreprocessedInstance.hf_tensor_fields, output_all_columns=True
        )
        return dataset


class VIMADataModuleFromLocalFiles(VIMADataModule):
    """VIMA DataModule which loads data from disk."""

    def __init__(self, *, dataset_data_dir: Path, **kwargs: Unpack[DataModuleKwargs]) -> None:
        super().__init__(**kwargs)
        self._dataset_data_dir = Path(dataset_data_dir)

    def _load_dataset(self) -> datasets.DatasetDict:
        """Load the dataset from the arrow files."""
        dataset = load_dataset_from_disk(
            self._dataset_data_dir,
            extension="arrow",
            config_name="preprocessed",
            num_proc=self._num_workers,
        )
        dataset = dataset.with_format(
            "torch", columns=PreprocessedInstance.hf_tensor_fields, output_all_columns=True
        )
        return dataset
