from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cogelot.common.config import flatten_config
from cogelot.common.hydra import instantiate_modules_from_hydra, preprocess_config_for_hydra

CONFIG_DIR = Path.cwd().joinpath("configs").as_posix()


@hydra.main(config_path=CONFIG_DIR, config_name="train.yaml", version_base="1.3")
def train_model(config: DictConfig) -> None:
    """Run the training."""
    config = preprocess_config_for_hydra(config)

    # Set the internal precision to be slightly lower than the default
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision("high")

    datamodule, model, trainer = instantiate_modules_from_hydra(config)

    logger.info("Saving hyperparameters...")
    model.save_hyperparameters(flatten_config(config))

    resume_from_checkpoint_path: str | None = OmegaConf.select(
        config, "resume_from_checkpoint", default=None
    )
    logger.info("Starting training...")
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_from_checkpoint_path)
    logger.info("Training finished.")


if __name__ == "__main__":
    train_model()
