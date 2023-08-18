from typing import Any, Literal

import datasets
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from cogelot.data.collate import collate_preprocessed_instances_from_hf_dataset
from cogelot.data.datasets import (
    download_parquet_files_from_hub,
    get_location_of_parquet_files,
    load_dataset_from_parquet_files,
    maybe_split_dataset_by_node,
    set_dataset_format,
)
from cogelot.data.evaluation import VIMAEvaluationDataset
from cogelot.structures.model import EvaluationEpisode, PreprocessedInstance


SetupStage = Literal["fit", "validate", "test", "predict"]


class VIMADataModule(LightningDataModule):
    """Datamodule for the VIMA dataset."""

    def __init__(
        self,
        *,
        hf_datasets_repo_name: str,
        num_workers: int,
        batch_size: int,
        dataloader_kwargs: dict[str, Any] | None = None
    ) -> None:
        super().__init__()
        self._hf_datasets_repo_name = hf_datasets_repo_name

        self._num_workers = num_workers
        self._dataloader_kwargs = dataloader_kwargs or {}

        self.batch_size = batch_size

        self.train_dataset: datasets.Dataset
        self.valid_dataset: datasets.Dataset
        self.eval_dataset: VIMAEvaluationDataset

    def prepare_data(self) -> None:
        """Prepare any data before starting training.

        This is just making sure the dataset has already been downloaded.
        """
        download_parquet_files_from_hub(self._hf_datasets_repo_name, max_workers=self._num_workers)

    def setup(self, stage: SetupStage) -> None:
        """Setup each GPU to run the data."""
        if stage == "fit":
            dataset = self._load_dataset()
            self.train_dataset = maybe_split_dataset_by_node(dataset["train"])
            self.valid_dataset = maybe_split_dataset_by_node(dataset["valid"])

        if stage == "validate":
            dataset = self._load_dataset()
            self.valid_dataset = maybe_split_dataset_by_node(dataset["valid"])

        if stage == "predict":
            self.eval_dataset = VIMAEvaluationDataset.from_partition_to_specs()

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

    def predict_dataloader(self) -> DataLoader[EvaluationEpisode]:
        """Get a dataloader to use for predicting during evaluation.

        Disable the `batch_size` and `batch_sampler` to ensure that we get a single instance at a time.
        """
        return DataLoader[EvaluationEpisode](
            self.eval_dataset,
            batch_size=None,
            shuffle=False,
            batch_sampler=None,
        )

    def _load_dataset(self) -> datasets.DatasetDict:
        """Load the dataset from the parquet files.

        This is not using `datasets.load_dataset` because doing it this separate way is much
        faster, but does have some complexity overhead.
        """
        dataset_data_dir = get_location_of_parquet_files(self._hf_datasets_repo_name)
        dataset = load_dataset_from_parquet_files(dataset_data_dir, num_proc=self._num_workers)
        dataset = set_dataset_format(dataset)
        return dataset
