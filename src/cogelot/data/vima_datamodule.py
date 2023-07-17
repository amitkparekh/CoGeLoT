from typing import Literal

import datasets
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from cogelot.data.datasets import dataloader_collate_fn, set_dataset_format
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
    ) -> None:
        super().__init__()
        self._hf_datasets_repo_name = hf_datasets_repo_name
        self._num_workers = num_workers

        self.batch_size = batch_size

        self.train_dataset: datasets.Dataset
        self.valid_dataset: datasets.Dataset

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
        return self._create_dataloader(self.train_dataset)

    def val_dataloader(self) -> DataLoader[list[PreprocessedInstance]]:
        """Create the dataloader for the validation set."""
        return self._create_dataloader(self.valid_dataset)

    def _create_dataloader(
        self, dataset: datasets.Dataset
    ) -> DataLoader[list[PreprocessedInstance]]:
        """Create a dataloader from a datapipe."""
        try:
            dataloader = DataLoader[list[PreprocessedInstance]](
                dataset,  # pyright: ignore[reportGeneralTypeIssues]
                batch_size=self.batch_size,
                num_workers=self._num_workers,
                shuffle=True,
                collate_fn=dataloader_collate_fn,
            )
        except (UnboundLocalError, AttributeError) as err:
            raise RuntimeError(
                "The dataset has not been initialized. Was setup() called?"
            ) from err

        return dataloader
