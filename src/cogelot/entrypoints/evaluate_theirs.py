import hydra
import torch
from loguru import logger
from omegaconf import DictConfig

from cogelot.common.config import flatten_config
from cogelot.common.hydra import instantiate_modules_from_hydra, preprocess_config_for_hydra
from cogelot.entrypoints.train import CONFIG_DIR


@hydra.main(config_path=CONFIG_DIR, config_name="evaluate_their_model.yaml", version_base="1.3")
def evaluate_model(config: DictConfig) -> None:
    """Run the evaluation."""
    # Set the internal precision to be slightly lower than the default
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision("high")

    config = preprocess_config_for_hydra(config)
    datamodule, model, trainer = instantiate_modules_from_hydra(config)

    logger.info("Saving hyperparameters...")
    model.save_hyperparameters(flatten_config(config))

    logger.info("Starting evaluation...")
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    evaluate_model()
