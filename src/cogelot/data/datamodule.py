import abc
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
from cogelot.common.settings import ConfigStage, DatasetVariant, Settings
from cogelot.data.collate import collate_preprocessed_instances_from_hf_dataset
from cogelot.data.datasets import only_select_indices_within_range, repeat_dataset_for_batch_size
from cogelot.data.evaluation import VIMAEvaluationDataset
from cogelot.structures.model import EvaluationEpisode, PreprocessedInstance
from cogelot.structures.vima import Task

SetupStage = Literal["fit", "validate", "test", "predict"]


class DataModuleKwargs(TypedDict):
    """Kwargs for the __init__ in the datamodule."""

    num_workers: int
    batch_size: int
    dataset_variant: DatasetVariant
    dataloader_kwargs: NotRequired[dict[str, Any]]

    # For filtering the dataset
    task_index_seen: NotRequired[int]
    dataset_start_index: NotRequired[int]
    max_num_instances_seen: NotRequired[int]


class VIMADataModule(abc.ABC, LightningDataModule):
    """Datamodule for the VIMA dataset."""

    _desired_config_stage: ConfigStage = "preprocessing"

    def __init__(self, **kwargs: Unpack[DataModuleKwargs]) -> None:
        super().__init__()
        self._kwargs = kwargs
        self._dataset_variant: DatasetVariant = kwargs["dataset_variant"]
        self._num_workers = kwargs.get("num_workers", 0)
        self.batch_size = kwargs.get("batch_size", 1)
        self._dataloader_kwargs = kwargs.get("dataloader_kwargs", {})

        self.train_dataset: datasets.Dataset
        self.valid_dataset: datasets.Dataset
        self.evaluation_dataset: VIMAEvaluationDataset

    def setup(self, stage: SetupStage) -> None:  # type: ignore[override]
        """Setup each node to run the data."""
        if stage == "fit":
            dataset = self._load_dataset()
            self.train_dataset = maybe_split_dataset_by_node(dataset["train"])
            self.valid_dataset = maybe_split_dataset_by_node(dataset["valid"])

        if stage == "validate":
            dataset = self._load_dataset()
            self.valid_dataset = maybe_split_dataset_by_node(dataset["valid"])

        if stage == "test":
            raise ValueError("Don't use this class for testing.")

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

    @abc.abstractmethod
    def _load_dataset(self) -> datasets.DatasetDict:
        """Load the dataset for all the splits."""
        raise NotImplementedError

    def _maybe_filter_datasets(self) -> None:
        """Apply filtering and such if desired."""
        max_num_instances_seen = self._kwargs.get("max_num_instances_seen")

        if max_num_instances_seen is not None:
            dataset_start_index = self._kwargs.get("dataset_start_index", 0)
            dataset_end_index = dataset_start_index + max_num_instances_seen

            logger.info(
                f"Selecting instances from `{dataset_start_index}` to `{dataset_end_index}` (not"
                " inclusive)"
            )

            self.train_dataset = repeat_dataset_for_batch_size(
                only_select_indices_within_range(
                    self.train_dataset, start=dataset_start_index, end=dataset_end_index
                ),
                batch_size=self.batch_size,
            )
            self.valid_dataset = repeat_dataset_for_batch_size(
                only_select_indices_within_range(
                    self.valid_dataset, start=dataset_start_index, end=dataset_end_index
                ),
                batch_size=self.batch_size,
            )


class VIMADataModuleFromHF(VIMADataModule):
    """VIMA DataModule by explicitly downloading the dataset from HF."""

    def __init__(self, *, hf_datasets_repo_name: str, **kwargs: Unpack[DataModuleKwargs]) -> None:
        super().__init__(**kwargs)
        self._hf_datasets_repo_name = hf_datasets_repo_name

    def prepare_data(self) -> None:
        """Prepare any data before starting training.

        This is just making sure the dataset has already been downloaded.
        """
        download_parquet_files_from_hub(
            self._hf_datasets_repo_name,
            pattern=f"{self._dataset_variant}/preprocessed*",
            max_workers=self._num_workers if self._num_workers > 0 else 1,
        )

    def _load_dataset(self) -> datasets.DatasetDict:
        """Load the dataset from the parquet files.

        This is not using `datasets.load_dataset` because doing it this separate way is much
        faster, but does have some complexity overhead.
        """
        task_index_seen = self._kwargs.get("task_index_seen")
        config_name = self._maybe_get_config_for_task_index(task_index_seen)

        dataset_data_dir = get_location_of_hub_parquet_files(self._hf_datasets_repo_name)
        dataset = load_dataset_from_disk(
            dataset_data_dir,
            extension="parquet",
            config_name=config_name,
            num_proc=self._num_workers,
        )
        dataset = dataset.with_format(
            "torch", columns=PreprocessedInstance.hf_tensor_fields, output_all_columns=True
        )
        return dataset

    def _maybe_get_config_for_task_index(self, task_index: int | None) -> str:
        """Get the config name for the task index, if the task index is provided."""
        settings = Settings()
        if not task_index:
            return settings.get_config_name(stage=self._desired_config_stage)

        task = Task(task_index)
        logger.info(f"Limiting task to `{task}`")
        return settings.get_config_name_for_task(task, stage=self._desired_config_stage)


class VIMADataModuleFromLocalFiles(VIMADataModule):
    """VIMA DataModule which loads data from disk."""

    def __init__(self, *, dataset_data_dir: Path, **kwargs: Unpack[DataModuleKwargs]) -> None:
        super().__init__(**kwargs)
        self._dataset_data_dir = Path(dataset_data_dir)

    def _load_dataset(self) -> datasets.DatasetDict:
        """Load the dataset from the arrow files."""
        task_index_seen = self._kwargs.get("task_index_seen")
        config_name: str = Task(task_index_seen).name if task_index_seen else ""

        dataset = load_dataset_from_disk(
            self._dataset_data_dir,
            extension="arrow",
            config_name=config_name,
            num_proc=self._num_workers,
        )
        dataset = dataset.with_format(
            "torch", columns=PreprocessedInstance.hf_tensor_fields, output_all_columns=True
        )
        return dataset


class VIMABenchOnlineDataModule(LightningDataModule):
    """Evaluation datamodule to run VIMA online."""

    def __init__(
        self,
        *,
        num_repeats_per_episode: int = 100,
        dataloader_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        super().__init__()
        self._num_repeats_per_episode = num_repeats_per_episode
        self._dataloader_kwargs = dataloader_kwargs

    def setup(self, stage: SetupStage) -> None:  # type: ignore[override]
        """Setup each node to run the data."""
        if stage == "test":
            self.evaluation_dataset = VIMAEvaluationDataset.from_partition_to_specs(
                num_repeats_per_episode=self._num_repeats_per_episode
            )
            return
        raise ValueError("Don't use this datamodule if you are not testing online.")

    def test_dataloader(self) -> DataLoader[EvaluationEpisode]:
        """Get a dataloader to use for creating environment during evaluation.

        Disable the `batch_size` and `batch_sampler` to ensure that we get a single instance at a time.
        """
        dataloader_kwargs: dict[str, Any] = {
            "batch_size": None,
            "shuffle": False,
            "batch_sampler": None,
            **self._dataloader_kwargs,
        }
        return DataLoader[EvaluationEpisode](self.evaluation_dataset, **dataloader_kwargs)
