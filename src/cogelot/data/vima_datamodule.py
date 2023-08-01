from typing import Any, Literal

import datasets
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from cogelot.data.collate import collate_preprocessed_instances_from_hf_dataset
from cogelot.data.datasets import set_dataset_format
from cogelot.structures.model import PreprocessedInstance


SetupStage = Literal["fit", "validate", "test", "predict"] | str


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

    def prepare_data(self) -> None:
        """Prepare any data before starting training.

        This is just making sure the dataset has already been downloaded.
        """
        datasets.load_dataset(self._hf_datasets_repo_name, num_proc=self._num_workers)

    def setup(self, stage: SetupStage) -> None:
        """Setup each GPU to run the data."""
        if stage == "fit":
            dataset = datasets.load_dataset(
                self._hf_datasets_repo_name, num_proc=self._num_workers
            )
            assert isinstance(dataset, datasets.DatasetDict)
            dataset = set_dataset_format(dataset)
            self.train_dataset = dataset["train"]
            self.valid_dataset = dataset["valid"]

        if stage == "validate":
            dataset = datasets.load_dataset(
                self._hf_datasets_repo_name, split="valid", num_proc=self._num_workers
            )
            assert isinstance(dataset, datasets.DatasetDict)
            dataset = set_dataset_format(dataset)
            self.valid_dataset = dataset["valid"]

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
