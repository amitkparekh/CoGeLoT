from typing import Annotated

import typer
from loguru import logger

from cogelot.common.hf_models import (
    download_model_checkpoint,
    get_model_checkpoint_file_in_remote_repo_for_epoch,
)


def download_checkpoint_from_hf(
    wandb_run_id: Annotated[str, typer.Argument()],
    *,
    hf_repo_id: Annotated[str, typer.Option()] = "amitkparekh/vima",
    epoch: Annotated[int, typer.Option()] = -1,
) -> None:
    """Download a model checkpoint from a wandb run in a Hugging Face repo."""
    model_path_in_repo = get_model_checkpoint_file_in_remote_repo_for_epoch(
        repo_id=hf_repo_id, run_id=wandb_run_id, epoch=epoch
    )
    logger.info(f"Downloading model from remote path: `{model_path_in_repo}`")
    model_checkpoint_path = download_model_checkpoint(
        repo_id=hf_repo_id, file_path_in_repo=model_path_in_repo
    )
    assert model_checkpoint_path.exists()
    logger.info(f"Model checkpoint downloaded to: `{model_checkpoint_path}`")


if __name__ == "__main__":
    typer.run(download_checkpoint_from_hf)
