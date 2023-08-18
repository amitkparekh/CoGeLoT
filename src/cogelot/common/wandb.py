from pathlib import Path

import wandb


def download_model_from_wandb(entity: str, project: str, run_id: str, save_dir: Path) -> Path:
    """Downlaod the model from wandb into the output dir and return the path to it."""
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    run_artifacts = run.logged_artifacts()
    assert len(run_artifacts) == 1, "There should only be one artifact in the run."

    model_artifact = run_artifacts[0]
    checkpoint_path = model_artifact.download(str(save_dir))

    return Path(checkpoint_path)
