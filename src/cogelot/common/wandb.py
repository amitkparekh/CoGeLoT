from pathlib import Path
from typing import Literal

from loguru import logger
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger


def download_model_from_wandb(
    entity: str, project: str, run_id: str, save_dir: Path, *, force_download: bool = False
) -> Path:
    """Downlaod the model from wandb into the output dir and return the path to it."""
    checkpoint_name = f"{entity}_{project}_{run_id}.ckpt"
    checkpoint_path = save_dir.joinpath(checkpoint_name)

    # If the checkpoint already exists, don't download it again
    if checkpoint_path.exists() and not force_download:
        return checkpoint_path

    # Get the run from wandb
    import wandb  # noqa: WPS433

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    # Get the model artifact from the run
    model_artifacts = [artifact for artifact in run.logged_artifacts() if artifact.type == "model"]
    assert len(model_artifacts) == 1, "There should only be one artifact in the run."

    # Download the model artifact (this can take a while)
    model_artifact_dir: str = model_artifacts[0].download(str(save_dir))

    # Rename the file to the expected name/location
    model_checkpoint_file = next(Path(model_artifact_dir).rglob("*.ckpt"))
    checkpoint_path = model_checkpoint_file.rename(checkpoint_path)

    return checkpoint_path


def get_id_from_current_run() -> str | None:
    """Get the current run ID from wandb, if there is a run in progress on the current machine."""
    import wandb  # noqa: WPS433

    if wandb.run is None:
        logger.warning(
            "Wandb is not initialized. Please initialize wandb before calling this function."
        )
        return None

    return wandb.run.id


class WandBWatchModelCallback(Callback):
    """When training with wandb, watch the model parameters and gradients."""

    def __init__(
        self,
        *,
        log: Literal["parameters", "gradients", "all"] = "all",
        log_freq: int = 1000,
        log_graph: bool = True,
    ) -> None:
        super().__init__()
        self._log = log
        self._log_freq = log_freq
        self._log_graph = log_graph

        self._wandb_logger: WandbLogger | None = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:  # noqa: ARG002
        """Setup the callback."""
        self._get_wandb_logger_from_trainer(trainer)
        if self._wandb_logger is None:
            logger.warning(
                "`WandbLogger` is not found within the trainer. Make sure you ware using WandB to log your experiments otherwise this callback will not work."
            )
            return

        logger.info("Watching model parameters and gradients.")
        self._wandb_logger.watch(
            pl_module,
            log=self._log,
            log_freq=self._log_freq,
            log_graph=self._log_graph,
        )

    def _get_wandb_logger_from_trainer(self, trainer: Trainer) -> None:
        """Get the WandbLogger from the trainer, if it exists."""
        if self._wandb_logger is None:
            for trainer_logger in trainer.loggers:
                if isinstance(trainer_logger, WandbLogger):
                    self._wandb_logger = trainer_logger
                    break
