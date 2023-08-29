from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Common settings used across the entrypoints."""

    seed: int = 1000

    # The number of instances to hold out for the validation set, across all the tasks.
    num_validation_instances: int = 50000

    # Things to help make the HF datasets go faster
    writer_batch_size: int = 10000

    # Maximum size for each shard that gets saved (by the HF dataset)
    max_shard_size: str = "1GB"

    # The repository ID on HF
    hf_repo_id: str = "amitkparekh/vima"
    raw_config_name: str = "raw"
    preprocessed_config_name: str = "preprocessed"

    # Directories
    storage_dir: Path = Path("storage/")
    storage_data_dir: Path = storage_dir.joinpath("data/")
    raw_data_dir: Path = storage_data_dir.joinpath("raw/vima_v6/")
    parsed_data_dir: Path = storage_data_dir.joinpath("parsed/")
    preprocessed_data_dir: Path = storage_data_dir.joinpath("preprocessed/")
    preprocessed_hf_dataset_dir: Path = storage_data_dir.joinpath("preprocessed_hf/")

    config_dir: Path = Path("configs/")
    instance_preprocessor_hydra_config: Path = config_dir.joinpath("instance_preprocessor.yaml")
