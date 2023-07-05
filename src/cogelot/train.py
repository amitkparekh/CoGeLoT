from typing import TYPE_CHECKING

import hydra
import lightning
from loguru import logger
from omegaconf import DictConfig


if TYPE_CHECKING:
    from lightning import pytorch as pl


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
def main(config: DictConfig) -> None:
    """Run the training."""
    seed = config.get("seed")
    if seed:
        lightning.seed_everything(seed)

    instantiated_modules = hydra.utils.instantiate(config)

    datamodule: pl.LightningDataModule = instantiated_modules["datamodule"]
    model: pl.LightningModule = instantiated_modules["model"]
    trainer: pl.Trainer = instantiated_modules["trainer"]

    logger.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
