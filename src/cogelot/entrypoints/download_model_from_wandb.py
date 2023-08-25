from pathlib import Path

from loguru import logger

from cogelot.common.wandb import download_model_from_wandb


def download_model(
    run_id: str,
    *,
    entity: str = "pyop",
    project: str = "CoGeLoT",
    save_dir: Path = Path("./storage/data/models"),
    force_download: bool = False,
) -> None:
    """Download the model checkpoint from wandb."""
    logger.info("Downloading model from wandb...")

    checkpoint_path = download_model_from_wandb(
        entity, project, run_id, save_dir, force_download=force_download
    )

    logger.info(f"Model downloaded to {checkpoint_path}")
