import sys
from pathlib import Path

import typer

from cogelot.common.hydra import (
    load_hydra_config,
    pretty_print_hydra_config,
    run_task_function_with_hydra,
)
from cogelot.entrypoints.create_hf_dataset import convert_vima_instance_to_hf_dataset
from cogelot.entrypoints.parse_raw_dataset import create_vima_instances_from_raw_dataset
from cogelot.entrypoints.run_models import evaluate_model, train_model


def override_sys_args_with_context(ctx: typer.Context) -> None:
    """Override the sys args with the context args.

    Override the args provides to the command by moving them all over. This is needed so that we
    can just call the various Hydra functions and all of the overrides (if they are used or not)
    are just automatically passed through by Hydra.
    """
    sys.argv[1:] = sys.argv[len(ctx.command_path.split(" ")) :]


app = typer.Typer(add_completion=False, no_args_is_help=True)

app.command("parse-raw-dataset")(create_vima_instances_from_raw_dataset)
app.command("create-hf-dataset")(convert_vima_instance_to_hf_dataset)


@app.command(context_settings={"allow_extra_args": True})
def print_config(config_file: Path, ctx: typer.Context) -> None:
    """Parse the hydra config file and pretty-print it."""
    overrides = ctx.args
    config = load_hydra_config(
        config_dir=config_file.parent, config_file_name=config_file.name, overrides=overrides
    )
    pretty_print_hydra_config(config)


@app.command(context_settings={"allow_extra_args": True})
def train(ctx: typer.Context, config_file: Path = Path("configs/train.yaml")) -> None:
    """Run the training loop for the model."""
    override_sys_args_with_context(ctx)

    run_task_function_with_hydra(
        config_dir=config_file.parent, config_file_name=config_file.name, task_function=train_model
    )


@app.command(context_settings={"allow_extra_args": True})
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
