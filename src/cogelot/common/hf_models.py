import os
from concurrent.futures import Future, wait
from pathlib import Path
from typing import Any, Literal, overload

from huggingface_hub import CommitInfo, HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from loguru import logger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from cogelot.common.wandb import get_id_from_current_run


def does_token_allow_write_access() -> bool:
    """Check if the token available to HF allows writing to the Hub."""
    api = HfApi()
    token_permission = api.get_token_permission()
    token_has_write_permission = token_permission == "write"  # noqa: S105

    logger.debug(f"HF token has write permission: {token_has_write_permission}")
    return token_has_write_permission


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


@overload
def upload_model_checkpoint(
    checkpoint_path: Path,
    run_id: str,
    repo_id: str,
    *,
    checkpoint_name: str | None = ...,
    run_as_future: Literal[False] = ...,
) -> None:
    ...


@overload
def upload_model_checkpoint(
    checkpoint_path: Path,
    run_id: str,
    repo_id: str,
    *,
    checkpoint_name: str | None = ...,
    run_as_future: Literal[True] = ...,
) -> Future[CommitInfo]:
    ...


def upload_model_checkpoint(
    checkpoint_path: Path,
    run_id: str,
    repo_id: str,
    *,
    checkpoint_name: str | None = None,
    run_as_future: bool = False,
) -> None | Future[CommitInfo]:
    """Upload a model checkpoint to the repository on the hub.

    If the run name is provied,

    If desired, you can provide the name of the checkpoint file. If not, it will use the name of
    the file that you are uploading.
    """
    create_model_repository(repo_id)

    if checkpoint_name is None:
        checkpoint_name = checkpoint_path.name

    api = HfApi()
    commit_info = api.upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo=f"{run_id}/{checkpoint_name}",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload {run_id}/{checkpoint_name}",
        run_as_future=run_as_future,  # pyright: ignore[reportGeneralTypeIssues]
    )
    if run_as_future:
        return commit_info
    return None


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


def get_model_checkpoint_file_in_remote_repo_for_epoch(
    repo_id: str, run_id: str, epoch: int
) -> str:
    """Get the filename of the checkpoint that we want to download, from the epoch.

    This is needed because the checkpoint filenames are not just the epoch number, and need
    parsing.
    """
    # Get all the files in the right folder
    api = HfApi()
    remote_files = api.list_files_info(repo_id=repo_id, paths=run_id, repo_type="model")
    all_file_names = [remote_file.path for remote_file in remote_files]

    # If epoch is negative, we want the last checkpoint
    if epoch < 0:
        epoch = len(all_file_names) - 1

    for file_name in all_file_names:
        if f"epoch={epoch:02}" in file_name:
            return file_name

    raise FileNotFoundError(
        f"Could not find a checkpoint for epoch {epoch} in the repo {repo_id}/{run_id}."
    )


def download_model_checkpoint(repo_id: str, run_id: str, file_path_in_repo: str) -> Path:
    """Download a model checkpoint from the repository on the hub."""
    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename=file_path_in_repo,
        subfolder=run_id,
        repo_type="model",
    )
    return Path(checkpoint_path)


class HuggingFaceModelLogger(Logger):
    """Logger for uploading models to the HuggingFace Hub.

    Currently, we are only uploading the best checkpoint for each run.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        use_hf_transfer: bool = False,
        upload_in_background_each_checkpoint: bool = False,
        experiment_id_override: str | None = None,
    ) -> None:
        super().__init__()

        self._repo_id = repo_id

        self._checkpoint_dir: Path | None = None
        self._checkpoint_callback: ModelCheckpoint | None = None
        self._experiment_id: str | None = experiment_id_override

        self._upload_in_background_each_checkpoint = upload_in_background_each_checkpoint
        self._upload_futures: list[Future] = []

        if use_hf_transfer:
            enable_hf_transfer()

        # Make sure that the token allows writing to the Hub, otherwise this logger should not be
        # used because it will fail and therefore the training will fail and we don't want that.
        if not does_token_allow_write_access():
            raise HfHubHTTPError("The token available to HF does not allow writing to the Hub.")

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

        if self._upload_in_background_each_checkpoint:
            self._upload_last_model_in_background(checkpoint_callback)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        """Upload the model to HF after the training has successfully finished."""
        if self._upload_futures:
            logger.info("Waiting for background uploads to finish...")
            wait(self._upload_futures)
            logger.info("Background uploads finished.")

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

        if not self._upload_in_background_each_checkpoint:
            logger.info(f"Uploading model checkpoints to `{self._repo_id}/{self._experiment_id}/`")
            upload_model_checkpoint_dir(self._checkpoint_dir, self._experiment_id, self._repo_id)

    def _upload_last_model_in_background(self, checkpoint_callback: ModelCheckpoint) -> None:
        """Upload the last model in the background."""
        if self._checkpoint_dir is None:
            logger.warning(
                "There is no checkpoint directory to upload. Does the ModelCheckpoint callback have a directory? Surely it must right?"
            )
            return
        if self._experiment_id is None:
            logger.error("There is no wandb run in progress, and we need one to group the models.")
            return

        upload_future = upload_model_checkpoint(  # pyright: ignore[reportGeneralTypeIssues]
            Path(checkpoint_callback._last_checkpoint_saved),  # noqa: SLF001
            run_id=self._experiment_id,
            repo_id=self._repo_id,
            run_as_future=True,
        )
        self._upload_futures.append(upload_future)
