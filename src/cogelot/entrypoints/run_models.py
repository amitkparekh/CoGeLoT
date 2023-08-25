from contextlib import suppress
from typing import NamedTuple

import hydra
import lightning
import torch
from lightning import pytorch as pl
from loguru import logger
from omegaconf import DictConfig

from cogelot.common import log


class InstantiatedModules(NamedTuple):
    """Tuple of the main instantiated modules from Hydra."""

    datamodule: pl.LightningDataModule
    model: pl.LightningModule
    trainer: pl.Trainer


def instantiate_modules_from_hydra(config: DictConfig) -> InstantiatedModules:
    """Instantiate the modules needed for training."""
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

    datamodule = hydra.utils.instantiate(config.datamodule)
    model = hydra.utils.instantiate(config.model)
    trainer = hydra.utils.instantiate(config.trainer, callbacks=model.callbacks)

    return InstantiatedModules(datamodule=datamodule, model=model, trainer=trainer)


def train_model(config: DictConfig) -> None:
    """Run the training."""
    datamodule, model, trainer = instantiate_modules_from_hydra(config)

    model.save_hyperparameters(log.flatten_config(config))

    logger.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)


def evaluate_model(config: DictConfig) -> None:
    """Run the evaluation."""
    datamodule, model, trainer = instantiate_modules_from_hydra(config)

    logger.info("Starting evaluation...")
    trainer.test(model, datamodule=datamodule)
