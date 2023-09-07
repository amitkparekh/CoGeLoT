import sys
from pathlib import Path

import typer

from cogelot.common.hydra import (
    load_hydra_config,
    pretty_print_hydra_config,
    run_task_function_with_hydra,
)
from cogelot.entrypoints.create_preprocessed_dataset_per_task import (
    create_preprocessed_dataset_per_task,
)
from cogelot.entrypoints.create_raw_dataset_per_task import create_raw_dataset_per_task
from cogelot.entrypoints.parse_original_dataset import parse_original_dataset
from cogelot.entrypoints.preprocess_instances import preprocess_instances
from cogelot.entrypoints.run_models import evaluate_model, train_model
from cogelot.entrypoints.settings import Settings
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
    overrides = sys.argv[len(ctx.command_path.split(" ")) :]
    sys.argv = [sys.argv[0], *overrides]


app = typer.Typer(add_completion=False, no_args_is_help=True)

app.command(rich_help_panel="Dataset Creation Commands")(parse_original_dataset)
app.command(rich_help_panel="Dataset Creation Commands")(create_raw_dataset_per_task)
app.command(rich_help_panel="Dataset Creation Commands")(preprocess_instances)
app.command(rich_help_panel="Dataset Creation Commands")(create_preprocessed_dataset_per_task)

app.command(rich_help_panel="Dataset Creation Commands")(upload_raw_dataset)
app.command(rich_help_panel="Dataset Creation Commands")(upload_preprocessed_dataset)


@app.command(context_settings={"allow_extra_args": True})
def print_config(config_file: Path, ctx: typer.Context) -> None:
    """Parse the hydra config file and pretty-print it."""
    overrides = ctx.args
    config = load_hydra_config(
        config_dir=config_file.parent, config_file_name=config_file.name, overrides=overrides
    )
    pretty_print_hydra_config(config)


@app.command(context_settings={"allow_extra_args": True}, rich_help_panel="Run Commands")
def train(ctx: typer.Context, config_file: Path = Path("configs/train.yaml")) -> None:
    """Run the training loop for the model."""
    override_sys_args_with_context(ctx)

    run_task_function_with_hydra(
        config_dir=config_file.parent, config_file_name=config_file.name, task_function=train_model
    )


@app.command(context_settings={"allow_extra_args": True}, rich_help_panel="Run Commands")
def evaluate(ctx: typer.Context, config_file: Path = Path("configs/evaluate.yaml")) -> None:
    """Run the evaluation loop for the model."""
    override_sys_args_with_context(ctx)

    run_task_function_with_hydra(
        config_dir=config_file.parent,
        config_file_name=config_file.name,
        task_function=evaluate_model,
    )


if __name__ == "__main__":
    app()
