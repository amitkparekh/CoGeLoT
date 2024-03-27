from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cogelot.common.hydra import instantiate_datamodule_from_hydra, preprocess_config_for_hydra

CONFIG_DIR = Path.cwd().joinpath("configs").as_posix()


@hydra.main(config_path=CONFIG_DIR, config_name="train.yaml", version_base="1.3")
def prepare_data(config: DictConfig) -> None:
    """Run the training."""
    logger.warning("Setting the trainer devices to 1.")
    OmegaConf.update(config, "trainer.devices", 1)

    config = preprocess_config_for_hydra(config)

    # Set the internal precision to be slightly lower than the default
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision("high")

    datamodule = instantiate_datamodule_from_hydra(config)

    logger.info("Preparing data...")
    datamodule.prepare_data()
    logger.info("Setting up data...")
    datamodule.setup("fit")
    logger.info("Done.")


if __name__ == "__main__":
    prepare_data()
