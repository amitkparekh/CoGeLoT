import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, create_repo, hf_hub_download
from loguru import logger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from cogelot.common.wandb import get_id_from_current_run


def enable_hf_transfer() -> None:
    """Enable the hf-transfer library for faster uploads/downloads.

    Reference:
        https://huggingface.co/docs/huggingface_hub/guides/download#faster-downloads
    """
    logger.warning(
        "HF's hf-transfer library makes things go so much faster, but you lose progress bars and a bunch of other nice things. It's also in a very early stage. Use at your own risk."
    )
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def create_model_repository(repo_id: str, *, is_private: bool = True) -> None:
    """Create a repository on the Hub for a model."""
    create_repo(repo_id, private=is_private, exist_ok=True)


def upload_model_checkpoint(
    checkpoint_path: Path, run_id: str, repo_id: str, *, checkpoint_name: str | None = None
) -> None:
    """Upload a model checkpoint to the repository on the hub.

    If the run name is provied,

    If desired, you can provide the name of the checkpoint file. If not, it will use the name of
    the file that you are uploading.
    """
    create_model_repository(repo_id)

    if checkpoint_name is None:
        checkpoint_name = checkpoint_path.name

    api = HfApi()
    api.upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo=f"{run_id}/{checkpoint_name}",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload {run_id}/{checkpoint_name}",
    )


def upload_model_checkpoint_dir(checkpoint_dir: Path, run_id: str, repo_id: str) -> None:
    """Upload all the checkpoints for a wandb run to the repository on the hub."""
    create_model_repository(repo_id)

    api = HfApi()
    api.upload_folder(
        repo_id=repo_id,
        path_in_repo=run_id,
        folder_path=checkpoint_dir,
        repo_type="model",
        commit_message=f"Upload checkpoints for {run_id}",
    )


def download_model_checkpoint(
    repo_id: str, run_id: str, checkpoint_name: str, output_dir: Path
) -> None:
    """Download a model checkpoint from the repository on the hub."""
    hf_hub_download(
        repo_id=repo_id,
        filename=checkpoint_name,
        subfolder=run_id,
        repo_type="model",
        local_dir=output_dir,
        local_dir_use_symlinks=False,
    )


class HuggingFaceModelLogger(Logger):
    """Logger for uploading models to the HuggingFace Hub.

    Currently, we are only uploading the best checkpoint for each run.
    """

    def __init__(self, repo_id: str, *, use_hf_transfer: bool = False) -> None:
        self._repo_id = repo_id

        self._checkpoint_dir: Path | None = None
        self._checkpoint_callback: ModelCheckpoint | None = None
        self._experiment_id: str | None = None

        if use_hf_transfer:
            enable_hf_transfer()

    @property
    def name(self) -> str | None:
        """Return the experiment name."""
        return None

    @property
    def version(self) -> str | None | int:
        """Return the experiment version."""
        return None

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:  # noqa: ARG002
        """Records metrics.

        This method logs metrics as soon as it received them.
        """
        return

    def log_hyperparams(self, params: dict[str, Any], *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        """Record hyperparameters."""
        return

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        """After save checkpoint is called, save the checkpoint callback and the experiment ID.

        This is so that we only need upload checkpoints once the training has finished.
        """
        if self._checkpoint_callback is None:
            self._checkpoint_callback = checkpoint_callback

        if not self._experiment_id:
            self._experiment_id = get_id_from_current_run()

        if not self._checkpoint_dir and checkpoint_callback.dirpath is not None:
            self._checkpoint_dir = Path(checkpoint_callback.dirpath)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        """Upload the model to HF after the training has successfully finished."""
        if status != "success":
            logger.info("Not uploading model because training was not successful.")
            return
        if self._checkpoint_callback is None:
            logger.warning(
                "There is no checkpoint callback to upload? Is the checkpoint callback included in the Trainer?"
            )
            return
        if not self._experiment_id:
            logger.warning(
                "There is no checkpoint callback or experiment ID to upload? This is the case if there is no wandb run associated."
            )
            return
        if not self._checkpoint_dir:
            logger.warning(
                "There is no checkpoint directory to upload. Does the ModelCheckpoint callback have a directory? Surely it must right?"
            )
            return

        logger.info(f"Uploading model checkpoints to `{self._repo_id}/{self._experiment_id}/`")
        upload_model_checkpoint_dir(self._checkpoint_dir, self._experiment_id, self._repo_id)
