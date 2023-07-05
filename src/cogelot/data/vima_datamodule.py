from pathlib import Path
from typing import Literal

from lightning import LightningDataModule
from torchdata.dataloader2 import (
    DataLoader2 as DataLoader,
    DistributedReadingService,
    MultiProcessingReadingService,
    SequentialReadingService,
)
from torchdata.datapipes.iter import IterDataPipe

from cogelot.data import datapipes
from cogelot.structures.model import PreprocessedInstance


SetupStage = Literal["fit", "validate", "test", "predict"] | str


class VIMADataModule(LightningDataModule):
    """Datamodule for the VIMA dataset."""

    def __init__(
        self,
        *,
        preprocessed_data_dir: Path,
        num_workers: int,
        batch_size: int,
        num_validation_instances: int,
    ) -> None:
        super().__init__()

        self._preprocessed_data_dir = Path(preprocessed_data_dir)

        self.batch_size = batch_size
        self._num_workers = num_workers

        self._num_validation_instances = num_validation_instances

        self.training_datapipe: IterDataPipe[list[PreprocessedInstance]]
        self.validation_datapipe: IterDataPipe[list[PreprocessedInstance]]

    def setup(self, stage: SetupStage) -> None:
        """Setup each GPU to run the data."""
        if stage == "fit":
            all_instances = datapipes.load_preprocessed_instances(self._preprocessed_data_dir)
            train_dataset, valid_dataset = datapipes.create_validation_split(
                all_instances, self._num_validation_instances
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
        mp_rs = MultiProcessingReadingService(num_workers=self._num_workers)
        dist_rs = DistributedReadingService()
        rs = SequentialReadingService(dist_rs, mp_rs)

        try:
            dl = DataLoader[list[PreprocessedInstance]](
                datapipe,  # pyright: ignore[reportGeneralTypeIssues]
                reading_service=rs,
            )
        except (UnboundLocalError, AttributeError) as err:
            raise RuntimeError(
                "The validation datapipe has not been initialized. Was setup() called?"
            ) from err

        return dl
