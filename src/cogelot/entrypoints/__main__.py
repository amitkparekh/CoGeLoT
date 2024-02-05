import sys
from pathlib import Path

import typer
from loguru import logger

from cogelot.common.hf_datasets import download_parquet_files_from_hub
from cogelot.common.hf_models import (
    download_model_checkpoint,
    get_model_checkpoint_file_in_remote_repo_for_epoch,
)
from cogelot.common.hydra import (
    load_hydra_config,
    pretty_print_hydra_config,
)
from cogelot.common.settings import Settings
from cogelot.entrypoints.create_preprocessed_dataset_per_task import (
    create_preprocessed_dataset_per_task,
)
from cogelot.entrypoints.create_raw_dataset_per_task import create_raw_dataset_per_task
from cogelot.entrypoints.create_reworded_dataset_per_task import (
    create_reworded_dataset_per_task,
)
from cogelot.entrypoints.fix_raw_dataset_per_task import fix_raw_dataset_per_task
from cogelot.entrypoints.parse_original_dataset import parse_original_dataset
from cogelot.entrypoints.preprocess_instances import preprocess_instances
from cogelot.entrypoints.upload_dataset import upload_preprocessed_dataset, upload_raw_dataset

settings = Settings()


def override_sys_args_with_context(ctx: typer.Context) -> None:
    """Override the sys args with the context args.

    Override the args provides to the command by moving them all over. This is needed so that we
    can just call the various Hydra functions and all of the overrides (if they are used or not)
    are just automatically passed through by Hydra.

    All the overrides need to be at index 1 and onwards. The first index is the command itself.
    Therefore anything between those indices are going to be lost. But it's okay, it's not too
    dire.
    """
    sys.argv = [sys.argv[0], *ctx.args]


app = typer.Typer(add_completion=False, no_args_is_help=True)

app.command(rich_help_panel="Dataset Creation Commands")(parse_original_dataset)
app.command(rich_help_panel="Dataset Creation Commands")(create_raw_dataset_per_task)
app.command(rich_help_panel="Dataset Creation Commands")(preprocess_instances)
app.command(rich_help_panel="Dataset Creation Commands")(create_preprocessed_dataset_per_task)
app.command(rich_help_panel="Dataset Creation Commands")(create_reworded_dataset_per_task)

app.command(rich_help_panel="Dataset Creation Commands")(upload_raw_dataset)
app.command(rich_help_panel="Dataset Creation Commands")(upload_preprocessed_dataset)

app.command(rich_help_panel="Dataset Creation Commands")(fix_raw_dataset_per_task)


@app.command(context_settings={"allow_extra_args": True})
def print_config(config_file: Path, ctx: typer.Context) -> None:
    """Parse the hydra config file and pretty-print it."""
    overrides = ctx.args
    config = load_hydra_config(
        config_dir=config_file.parent, config_file_name=config_file.name, overrides=overrides
    )
    pretty_print_hydra_config(config)


@app.command(rich_help_panel="Downloading")
def download_training_data(
    num_workers: int = 0,
    hf_datasets_repo_name: str = "amitkparekh/vima",
) -> None:
    """Download the training data from HF in advance.

    In can take a while when not using loads of workers for parallel downloads, which is the case
    during the training command. So, this is a good way to do it quickly.
    """
    download_parquet_files_from_hub(
        hf_datasets_repo_name, pattern="**/preprocessed*/**", max_workers=num_workers
    )


@app.command(rich_help_panel="Downloading")
def download_model(
    wandb_run_id: str, hf_repo_id: str = "amitkparekh/vima", epoch: int = -1
) -> None:
    """Download the model checkpoint from HF."""
    model_path_in_repo = get_model_checkpoint_file_in_remote_repo_for_epoch(
        repo_id=hf_repo_id, run_id=wandb_run_id, epoch=epoch
    )
    logger.info(f"Downloading model from remote path: `{model_path_in_repo}`")
    model_checkpoint_path = download_model_checkpoint(
        repo_id=hf_repo_id, file_path_in_repo=model_path_in_repo
    )
    logger.info(f"Model downloaded to {model_checkpoint_path}")


if __name__ == "__main__":
    app()
