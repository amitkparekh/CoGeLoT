from contextlib import suppress
from typing import TYPE_CHECKING

import hydra
import lightning
import torch
from loguru import logger
from omegaconf import DictConfig


if TYPE_CHECKING:
    from lightning import pytorch as pl


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="evaluate.yaml")
def main(config: DictConfig) -> None:
    """Run the evaluation."""
    seed = config.get("seed")
    if seed:
        lightning.seed_everything(seed)

    # Try to set the sharing sharing strategy to file system, but don't fail if it's not supported.
    # This is an alternative to running `ulimit -S -n unlimited` in the shell.
    with suppress(AssertionError):
        torch.multiprocessing.set_sharing_strategy("file_system")

    logger.info("Instantiating modules...")
    instantiated_modules = hydra.utils.instantiate(config)

    datamodule: pl.LightningDataModule = instantiated_modules["datamodule"]
    model: pl.LightningModule = instantiated_modules["model"]
    trainer: pl.Trainer = instantiated_modules["trainer"]

    logger.info("Starting evaluation...")
    trainer.test(model, datamodule=datamodule)
