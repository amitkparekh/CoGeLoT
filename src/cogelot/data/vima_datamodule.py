from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from lightning import LightningDataModule
from loguru import logger
from torchdata.dataloader2 import (
    DataLoader2 as DataLoader,
    MultiProcessingReadingService,
)
from torchdata.datapipes.iter import IterDataPipe

from cogelot.data import datapipes
from cogelot.structures.model import PreprocessedInstance


if TYPE_CHECKING:
    from pathlib import Path

    from cogelot.data.preprocess import InstancePreprocessor


SetupStage = Literal["fit", "validate", "test", "predict"] | str


class VIMADataModule(LightningDataModule):
    """Datamodule for the VIMA dataset."""

    def __init__(
        self,
        *,
        instance_preprocessor: InstancePreprocessor,
        raw_data_dir: Path,
        normalized_data_dir: Path,
        preprocessed_data_dir: Path,
        num_workers: int,
        batch_size: int,
    ) -> None:
        super().__init__()

        self.instance_preprocessor = instance_preprocessor

        self.raw_data_dir = raw_data_dir
        self.normalized_data_dir = normalized_data_dir
        self.preprocessed_data_dir = preprocessed_data_dir

        self.batch_size = batch_size
        self._num_workers = num_workers

        self.training_datapipe: IterDataPipe[list[PreprocessedInstance]]

    def prepare_data(self) -> None:
        """Perform any downloading and preparing of the dataset.

        This will only happen once.
        """
        # TODO: Download the dataset
        # TODO: Extract the dataset

        # Normalize the raw dataset
        normalize_raw_datapipe = datapipes.normalize_raw_data(self.raw_data_dir)
        # Save the normalized dataset to disk
        cached_normalized_datapipe = datapipes.cache_normalized_data(
            normalize_raw_datapipe, self.normalized_data_dir
        )

        # Load the normalized data from disk
        loaded_normalized_datapipe = datapipes.load_cached_normalized_data(
            self.normalized_data_dir
        )
        # Preprocess the normalized dataset (e.g. tokenize, etc.)
        preprocessed_datapipe = loaded_normalized_datapipe.map(self.instance_preprocessor.process)
        # Cache the preprocessed dataset to disk
        cached_preprocessed_datapipe = datapipes.cache_preprocessed_data(
            preprocessed_datapipe, self.preprocessed_data_dir
        )

        # Run the pipelines
        logger.info("Normalizing and saving raw data")
        list(
            DataLoader(
                cached_normalized_datapipe,  # pyright: ignore[reportGeneralTypeIssues]
                reading_service=MultiProcessingReadingService(num_workers=self._num_workers),
            )
        )

        logger.info("Preprocessing and saving normalized data")
        list(
            DataLoader(
                cached_preprocessed_datapipe,  # pyright: ignore[reportGeneralTypeIssues]
                reading_service=MultiProcessingReadingService(num_workers=self._num_workers),
            )
        )

        logger.info("Data preparation complete!")

    def setup(self, stage: SetupStage) -> None:
        """Setup each GPU to run the data."""
        if stage == "fit":
            self.training_datapipe = cast(
                IterDataPipe[list[PreprocessedInstance]],
                datapipes.load_cached_preprocessed_data(self.preprocessed_data_dir)
                .shuffle()
                .sharding_filter()
                .batch(self.batch_size, drop_last=True),
            )

    def train_dataloader(self) -> DataLoader[list[PreprocessedInstance]]:
        """Create the dataloader for the training set."""
        mp_rs = MultiProcessingReadingService(num_workers=self._num_workers)

        try:
            dl = DataLoader[list[PreprocessedInstance]](
                self.training_datapipe,  # pyright: ignore[reportGeneralTypeIssues]
                reading_service=mp_rs,
            )
        except (UnboundLocalError, AttributeError) as err:
            raise RuntimeError(
                "The training datapipe has not been initialized. Was setup() called?"
            ) from err

        return dl
