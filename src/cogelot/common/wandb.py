from pathlib import Path

from loguru import logger


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
    import wandb

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
    import wandb

    if wandb.run is None:
        logger.warning(
            "Wandb is not initialized. Please initialize wandb before calling this function."
        )
        return None

    return wandb.run.id
