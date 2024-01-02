from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings

from cogelot.structures.vima import Task

DATASET_VARIANT = Literal["original", "keep_null_action"]
CONFIG_STAGES = Literal["parsing", "preprocessing"]


class Settings(BaseSettings):
    """Common settings used across the entrypoints."""

    seed: int = 1000

    # The number of instances to hold out for the validation set, across all the tasks.
    num_validation_instances: int = 50000

    # Max number of shards to use when saving/uploading the HF dataset
    num_train_shards: int = 20
    num_valid_shards: int = 5

    # The repository ID on HF
    hf_repo_id: str = "amitkparekh/vima"
    parsed_config_name: str = "parsed"
    preprocessed_config_name: str = "preprocessed"

    dataset_variant: DATASET_VARIANT = "original"

    # Directories
    storage_dir: Path = Path("storage/")
    storage_data_dir: Path = storage_dir.joinpath("data/")
    raw_data_dir: Path = storage_data_dir.joinpath("raw/vima_v6/")

    _instances_subdir: str = "instances/"
    _hf_subdir: str = "hf/"
    _hf_parquets_subdir: str = "hf_parquets/"

    config_dir: Path = Path("configs/")
    instance_preprocessor_hydra_config: Path = config_dir.joinpath("instance_preprocessor.yaml")

    @property
    def safe_hf_repo_id(self) -> str:
        """Return file-safe HF repo id.

        Basically, replace the `/` with `--`.
        """
        return self.hf_repo_id.replace("/", "--")

    @property
    def num_shards(self) -> dict[str, int]:
        """Get the number of shards per split."""
        return {"train": self.num_train_shards, "valid": self.num_valid_shards}

    @property
    def parsed_data_dir(self) -> Path:
        """Location of all parsed data."""
        return self.storage_data_dir.joinpath(self.dataset_variant, self.parsed_config_name)

    @property
    def parsed_instances_dir(self) -> Path:
        """Location of all parsed instances."""
        return self.parsed_data_dir.joinpath(self._instances_subdir)

    @property
    def parsed_hf_dataset_dir(self) -> Path:
        """Location of the arrow files for the HF dataset."""
        return self.parsed_data_dir.joinpath(self._hf_subdir)

    @property
    def preprocessed_data_dir(self) -> Path:
        """Location of all preprocessed data."""
        return self.storage_data_dir.joinpath(self.dataset_variant, self.preprocessed_config_name)

    @property
    def preprocessed_instances_dir(self) -> Path:
        """Location of all preprocessed instances."""
        return self.preprocessed_data_dir.joinpath(self._instances_subdir)

    @property
    def preprocessed_hf_dataset_dir(self) -> Path:
        """Location of the arrow files for the HF dataset."""
        return self.preprocessed_data_dir.joinpath(self._hf_subdir)

    @property
    def hf_parquets_dir(self) -> Path:
        """Location of the parquet files for the HF dataset."""
        return self.storage_data_dir.joinpath(self._hf_parquets_subdir)

    def get_config_name(self, *, stage: CONFIG_STAGES) -> str:
        """Get the config name for the given stage."""
        name_for_stage = {
            "parsing": self.parsed_config_name,
            "preprocessing": self.preprocessed_config_name,
        }
        return f"{self.dataset_variant}/{name_for_stage[stage]}"

    def get_config_name_for_task(self, task: Task, *, stage: CONFIG_STAGES) -> str:
        """Get the config name for the given task."""
        return f"{self.get_config_name(stage=stage)}--{task.name}"
