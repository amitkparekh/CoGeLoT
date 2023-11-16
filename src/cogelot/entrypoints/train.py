from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig

from cogelot.common.config import flatten_config
from cogelot.common.hydra import instantiate_modules_from_hydra
from cogelot.data.datamodule import VIMADataModule

CONFIG_DIR = Path.cwd().joinpath("configs").as_posix()


@hydra.main(config_path=CONFIG_DIR, config_name="train.yaml", version_base="1.3")
def train_model(config: DictConfig) -> None:
    """Run the training."""
    # Set the internal precision to be slightly lower than the default
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision("high")

    datamodule, model, trainer = instantiate_modules_from_hydra(config)

    model.save_hyperparameters(flatten_config(config))

    logger.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)

    logger.info("Changing batch size in datamodule to 1 for testing...")
    assert isinstance(datamodule, VIMADataModule)
    datamodule.batch_size = 1

    logger.info("Starting testing...")
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    train_model()
