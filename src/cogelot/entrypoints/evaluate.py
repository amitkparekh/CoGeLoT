import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cogelot.common.config import flatten_config
from cogelot.common.config_metadata_patcher import build_eval_run_name, update_eval_config
from cogelot.common.hydra import instantiate_modules_from_hydra, preprocess_config_for_hydra
from cogelot.entrypoints.train import CONFIG_DIR
from cogelot.models.evaluation import EvaluationLightningModule


@hydra.main(config_path=CONFIG_DIR, config_name="evaluate.yaml", version_base="1.3")
def evaluate_model(config: DictConfig) -> None:
    """Run the evaluation."""
    # Set the internal precision to be slightly lower than the default
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision("high")

    config = update_eval_config(config)
    OmegaConf.update(
        config, "trainer.logger.wandb.name", build_eval_run_name(config), force_add=True
    )

    config = preprocess_config_for_hydra(config)

    datamodule, model, trainer = instantiate_modules_from_hydra(config)

    logger.info("Saving hyperparameters...")
    model.save_hyperparameters(flatten_config(config))

    # Patch the transformer decoder to enable greedy token generation
    assert isinstance(model, EvaluationLightningModule)
    model.eval()
    if model.model.policy.num_action_tokens_per_timestep > 1:
        model.model.policy.use_greedy_decoding = True

    logger.info("Starting evaluation...")
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    evaluate_model()
