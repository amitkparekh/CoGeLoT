from loguru import logger
from omegaconf import DictConfig

from cogelot.common.hydra import instantiate_modules_from_hydra


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
