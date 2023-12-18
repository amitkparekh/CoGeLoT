from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings

DATASET_VARIANT = Literal["original"]


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
    raw_config_name: str = "raw"
    preprocessed_config_name: str = "preprocessed"

    dataset_variant: DATASET_VARIANT = "original"

    # Directories
    storage_dir: Path = Path("storage/")
    storage_data_dir: Path = storage_dir.joinpath("data/")
    raw_data_dir: Path = storage_data_dir.joinpath("raw/vima_v6/")

    parsed_data_dir: Path = storage_data_dir.joinpath(dataset_variant, "parsed/")
    parsed_instances_dir: Path = parsed_data_dir.joinpath("instances/")
    parsed_hf_dataset_dir: Path = parsed_data_dir.joinpath("hf/")
    parsed_hf_parquets_dir: Path = parsed_data_dir.joinpath("hf_parquets/")

    preprocessed_data_dir: Path = storage_data_dir.joinpath(dataset_variant, "preprocessed/")
    preprocessed_instances_dir: Path = preprocessed_data_dir.joinpath("instances/")
    preprocessed_hf_dataset_dir: Path = preprocessed_data_dir.joinpath("hf/")
    preprocessed_hf_parquets_dir: Path = preprocessed_data_dir.joinpath("hf_parquets/")

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
