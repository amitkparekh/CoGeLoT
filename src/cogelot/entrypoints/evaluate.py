import torch
from loguru import logger
from omegaconf import DictConfig

from cogelot.common.hydra import instantiate_modules_from_hydra, preprocess_config_for_hydra


def validate_model(config: DictConfig) -> None:
    """Run the evaluation."""
    # Set the internal precision to be slightly lower than the default
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision("high")

    config = preprocess_config_for_hydra(config)
    datamodule, model, trainer = instantiate_modules_from_hydra(config)

    logger.info("Starting validation...")
    trainer.validate(model, datamodule=datamodule)


def evaluate_model(config: DictConfig) -> None:
    """Run the evaluation."""
    # Set the internal precision to be slightly lower than the default
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision("high")

    config = preprocess_config_for_hydra(config)
    datamodule, model, trainer = instantiate_modules_from_hydra(config)

    logger.info("Starting evaluation...")
    trainer.test(model, datamodule=datamodule)
