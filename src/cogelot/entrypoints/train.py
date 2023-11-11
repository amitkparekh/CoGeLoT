import hydra
from loguru import logger
from omegaconf import DictConfig

from cogelot.common.config import flatten_config
from cogelot.common.hydra import instantiate_modules_from_hydra


@hydra.main(config_path="configs", config_name="train.yaml")
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


if __name__ == "__main__":
    train_model()
