from pathlib import Path
from typing import Any, Literal

import datasets
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from cogelot.data.collate import collate_preprocessed_instances_from_hf_dataset
from cogelot.data.datasets import (
    download_parquet_files_from_hub,
    load_dataset_from_parquet_files,
    maybe_split_dataset_by_node,
    set_dataset_format,
)
from cogelot.structures.model import PreprocessedInstance


SetupStage = Literal["fit", "validate", "test", "predict"] | str


class VIMADataModule(LightningDataModule):
    """Datamodule for the VIMA dataset."""

    def __init__(
        self,
        *,
        hf_datasets_repo_name: str,
        dataset_data_dir: Path,
        num_workers: int,
        batch_size: int,
        dataloader_kwargs: dict[str, Any] | None = None
    ) -> None:
        super().__init__()
        self._hf_datasets_repo_name = hf_datasets_repo_name
        self._dataset_data_dir = dataset_data_dir.joinpath(
            self._hf_datasets_repo_name.replace("/", "__")
        )

        self._num_workers = num_workers
        self._dataloader_kwargs = dataloader_kwargs or {}

        self.batch_size = batch_size

        self.train_dataset: datasets.Dataset
        self.valid_dataset: datasets.Dataset

    def prepare_data(self) -> None:
        """Prepare any data before starting training.

        This is just making sure the dataset has already been downloaded.
        """
        self._dataset_data_dir.mkdir(parents=True, exist_ok=True)
        download_parquet_files_from_hub(
            self._hf_datasets_repo_name,
            output_dir=self._dataset_data_dir,
            max_workers=self._num_workers,
        )

    def setup(self, stage: SetupStage) -> None:
        """Setup each GPU to run the data."""
        if stage == "fit":
            dataset = self._load_dataset()
            self.train_dataset = maybe_split_dataset_by_node(dataset["train"])
            self.valid_dataset = maybe_split_dataset_by_node(dataset["valid"])

        if stage == "validate":
            dataset = self._load_dataset()
            self.valid_dataset = maybe_split_dataset_by_node(dataset["valid"])

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

    def _load_dataset(self) -> datasets.DatasetDict:
        """Load the dataset from HF."""
        dataset = load_dataset_from_parquet_files(
            self._dataset_data_dir, num_proc=self._num_workers
        )
        dataset = set_dataset_format(dataset)
        return dataset
