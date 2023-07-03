from pathlib import Path
from typing import Literal

from lightning import LightningDataModule
from loguru import logger
from torchdata.dataloader2 import (
    DataLoader2 as DataLoader,
    MultiProcessingReadingService,
)
from torchdata.datapipes.iter import IterDataPipe

from cogelot.data import datapipes
from cogelot.data.preprocess import InstancePreprocessor
from cogelot.structures.model import PreprocessedInstance


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
        num_validation_instances: int
    ) -> None:
        super().__init__()

        self.instance_preprocessor = instance_preprocessor

        self._raw_data_dir = raw_data_dir
        self._normalized_data_dir = normalized_data_dir
        self._preprocessed_data_dir = preprocessed_data_dir

        self.batch_size = batch_size
        self._num_workers = num_workers

        self._num_validation_instances = num_validation_instances

        self.training_datapipe: IterDataPipe[list[PreprocessedInstance]]
        self.validation_datapipe: IterDataPipe[list[PreprocessedInstance]]

    def prepare_data(self) -> None:
        """Perform any downloading and preparing of the dataset.

        This will only happen once.
        """
        # TODO: Download the dataset
        # TODO: Extract the dataset

        # Normalize the raw dataset
        normalize_raw_datapipe = datapipes.normalize_raw_data(self._raw_data_dir)
        # Save the normalized dataset to disk
        cached_normalized_datapipe = datapipes.cache_normalized_data(
            normalize_raw_datapipe, self._normalized_data_dir
        )

        # Load the normalized data from disk
        loaded_normalized_datapipe = datapipes.load_cached_normalized_data(
            self._normalized_data_dir
        )
        # Preprocess the normalized dataset (e.g. tokenize, etc.)
        preprocessed_datapipe = loaded_normalized_datapipe.map(
            self.instance_preprocessor.preprocess
        )
        # Cache the preprocessed dataset to disk
        cached_preprocessed_datapipe = datapipes.cache_preprocessed_data(
            preprocessed_datapipe, self._preprocessed_data_dir
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
            train_dataset, valid_dataset = datapipes.create_validation_split(
                datapipes.load_cached_preprocessed_data(self._preprocessed_data_dir),
                self._num_validation_instances,
            )

            self.training_datapipe = datapipes.batch_datapipe(train_dataset, self.batch_size)
            self.validation_datapipe = datapipes.batch_datapipe(valid_dataset, self.batch_size)

    def train_dataloader(self) -> DataLoader[list[PreprocessedInstance]]:
        """Create the dataloader for the training set."""
        return self._create_dataloader(self.training_datapipe)

    def val_dataloader(self) -> DataLoader[list[PreprocessedInstance]]:
        """Create the dataloader for the validation set."""
        return self._create_dataloader(self.validation_datapipe)

    def _create_dataloader(
        self, datapipe: IterDataPipe[list[PreprocessedInstance]]
    ) -> DataLoader[list[PreprocessedInstance]]:
        """Create a dataloader from a datapipe."""
        try:
            dl = DataLoader[list[PreprocessedInstance]](
                datapipe,  # pyright: ignore[reportGeneralTypeIssues]
                reading_service=MultiProcessingReadingService(num_workers=self._num_workers),
            )
        except (UnboundLocalError, AttributeError) as err:
            raise RuntimeError(
                "The validation datapipe has not been initialized. Was setup() called?"
            ) from err

        return dl
