from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

from cogelot.common.config import flatten_config
from cogelot.common.hydra import instantiate_modules_from_hydra

CONFIG_DIR = Path.cwd().joinpath("configs").as_posix()


def send_config_to_wandb(config: DictConfig) -> None:
    """Send the config to wandb using the wandb module."""
    import wandb

    wandb.config.update(flatten_config(config))
    wandb.config.update()


@hydra.main(config_path=CONFIG_DIR, config_name="train.yaml", version_base="1.3")
def train_model(config: DictConfig) -> None:
    """Run the training."""
    datamodule, model, trainer = instantiate_modules_from_hydra(config)

    model.save_hyperparameters(flatten_config(config))
    send_config_to_wandb(config)

    logger.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train_model()
