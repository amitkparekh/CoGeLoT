from pathlib import Path

import wandb


def download_model_from_wandb(
    entity: str, project: str, run_id: str, save_dir: Path, *, force_download: bool = False
) -> Path:
    """Downlaod the model from wandb into the output dir and return the path to it."""
    checkpoint_name = f"{entity}_{project}_{run_id}.ckpt"
    checkpoint_path = save_dir.joinpath(checkpoint_name)

    if checkpoint_path.exists() and not force_download:
        return checkpoint_path

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    run_artifacts = run.logged_artifacts()
    model_artifacts = [artifact for artifact in run_artifacts if artifact.type == "model"]
    assert len(model_artifacts) == 1, "There should only be one artifact in the run."

    checkpoint_path = model_artifacts[0].download(str(checkpoint_path))

    return Path(checkpoint_path)
