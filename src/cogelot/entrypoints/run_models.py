from contextlib import suppress
from typing import NamedTuple

import hydra
import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig

from cogelot.common.config import flatten_config


class InstantiatedModules(NamedTuple):
    """Tuple of the main instantiated modules from Hydra."""

    datamodule: pl.LightningDataModule
    model: pl.LightningModule
    trainer: pl.Trainer


def instantiate_modules_from_hydra(config: DictConfig) -> InstantiatedModules:
    """Instantiate the modules needed for training."""
    seed = config.get("seed")
    if seed:
        pl.seed_everything(seed)

    # Try to set the sharing sharing strategy to file system, but don't fail if it's not supported.
    # This is an alternative to running `ulimit -S -n unlimited` in the shell.
    with suppress(AssertionError):
        torch.multiprocessing.set_sharing_strategy("file_system")

    logger.info("Instantiating modules...")
    instantiated_modules = hydra.utils.instantiate(config)

    datamodule: pl.LightningDataModule = instantiated_modules["datamodule"]
    model: pl.LightningModule = instantiated_modules["model"]
    trainer: pl.Trainer = instantiated_modules["trainer"]

    return InstantiatedModules(datamodule=datamodule, model=model, trainer=trainer)


def train_model(config: DictConfig) -> None:
    """Run the training."""
    datamodule, model, trainer = instantiate_modules_from_hydra(config)

    # Manually update the hparams with the flattened config, since calling `save_hyperparameters`
    # wants to follow all of the frame stack, which then causing breaking as I'm using typer to run
    # things. So I've made things harder for myself by wanting to use a library.
    model._log_hyperparams = True  # noqa: SLF001
    model._set_hparams(flatten_config(config))  # noqa: SLF001
    for model_logger in model.loggers:
        model_logger.log_hyperparams(flatten_config(config))

    logger.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)


def validate_model(config: DictConfig) -> None:
    """Run the evaluation."""
    datamodule, model, trainer = instantiate_modules_from_hydra(config)

    logger.info("Starting validation...")
    trainer.validate(model, datamodule=datamodule)


def evaluate_model(config: DictConfig) -> None:
    """Run the evaluation."""
    datamodule, model, trainer = instantiate_modules_from_hydra(config)

    logger.info("Starting evaluation...")
    trainer.test(model, datamodule=datamodule)
