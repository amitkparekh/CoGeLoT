from pathlib import Path
from typing import Annotated, Optional

import click
import hydra
import torch
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from cogelot.common.hydra import load_hydra_config
from cogelot.data.evaluation import get_every_partition_task_combination
from cogelot.models.evaluation import EvaluationLightningModule
from cogelot.structures.vima import Partition, Task

all_evaluation_episodes = get_every_partition_task_combination()
map_task_to_partitions = {
    task: [episode.partition for episode in all_evaluation_episodes if episode.task == task]
    for task in Task
}
console = Console()


def create_evaluation_module(config_path: Path) -> EvaluationLightningModule:
    """Create the evaluation lightning module."""
    config = load_hydra_config(
        config_path.parent,
        config_path.name,
        overrides=["model.model.wandb_run_id=8lkml12g", "environment@model.environment=display"],
    )
    evaluation = hydra.utils.instantiate(config["model"])
    assert isinstance(evaluation, EvaluationLightningModule)
    return evaluation


def _print_task_table() -> Table:
    table = Table()
    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Task", style="magenta")
    table.add_column("Partitions", style="green")

    for task in Task:
        formatted_partitions = []
        for partition in Partition:
            if partition not in map_task_to_partitions[task]:
                formatted_partitions.append(f"[dim]L{partition.value}[/dim]")
            else:
                formatted_partitions.append(f"L{partition.value}")
        table.add_row(
            str(task.value + 1),
            task.name,
            " ".join(formatted_partitions),
        )
    return table


def _parse_task(task: int | str | Task | None) -> Task:
    if isinstance(task, Task):
        return task
    if isinstance(task, str):
        task = int(task) if task.isnumeric() else Task[task]
    if isinstance(task, int):
        return Task(task - 1)
    raise ValueError(f"Invalid task: {task}")


def _parse_partition(partition: int | str | Partition | None, *, task: Task) -> Partition:
    if partition is None:
        raise ValueError("Partition cannot be None")
    if isinstance(partition, str):
        partition = int(partition) if partition.isnumeric() else Partition[partition]
    if isinstance(partition, int) or (isinstance(partition, str) and partition.isnumeric()):
        partition = Partition(int(partition))
    if partition not in map_task_to_partitions[task]:
        raise ValueError(f"Invalid partition `{partition}` for task `{task}`")
    return partition


app = typer.Typer(add_completion=False)


class TaskChoices(click.Choice):
    """Click choice for picking a task."""

    name = "TaskChoices"

    def __init__(self) -> None:
        names = [task.name for task in Task]
        numbers = [str(task.value + 1) for task in Task]
        super().__init__([*names, *numbers])

    def convert(
        self,
        value: str | int | Task | None,
        param: click.Parameter | None,  # noqa: ARG002
        ctx: click.Context | None,  # noqa: ARG002
    ) -> Task | None:
        """Convert the input into a task, if possible."""
        if not value:
            return None
        return _parse_task(value)


class PartitionChoices(click.Choice):
    """Click choice for picking the partition."""

    name = "PartitionChoices"

    def __init__(self) -> None:
        names = [partition.name for partition in Partition]
        numbers = [str(partition.value) for partition in Partition]
        super().__init__([*names, *numbers])

    def convert(
        self,
        value: str | int | Partition | None,
        param: click.Parameter | None,  # noqa: ARG002
        ctx: click.Context | None,
    ) -> Partition | None:
        """Convert the input into a partition, if possible."""
        if not value:
            return None
        if ctx is not None:
            task = ctx.params["task"]
            return _parse_partition(value, task=task)
        return Partition.placement_generalization


@app.command()
def interactive_evaluate(
    task: Annotated[
        Optional[str],  # noqa: UP007
        typer.Option(click_type=TaskChoices(), help="Pick the task."),
    ] = None,
    partition: Annotated[
        Optional[str],  # noqa: UP007
        typer.Option(click_type=PartitionChoices(), help="Pick the partition."),
    ] = None,
) -> None:
    """Run the evalaution interactively, so we can see what's happening.

    Display is required for this.
    """
    console.print(_print_task_table())

    # If the task is not specified, ask the user to pick one
    if not isinstance(task, Task):
        task = typer.prompt(
            "Pick a task, either by name or number",
            type=TaskChoices(),
            show_choices=False,
        )
        assert isinstance(task, Task)

    if isinstance(partition, Partition) and partition not in map_task_to_partitions[task]:
        logger.error("Invalid partition for task.")
        partition = None

    if not isinstance(partition, Partition):
        partition = typer.prompt("Pick a partition", type=PartitionChoices())
        assert isinstance(partition, Partition)

    logger.info(f"Running {task} on {partition}")

    # Create the evaluation lightning module
    evaluation = create_evaluation_module(Path("configs/evaluate.yaml"))

    # Reset the environment with the task and partition
    evaluation.reset_environment(task=task, partition=partition)

    # Create the VIMA instance from the environment
    vima_instance = evaluation.environment.create_vima_instance(partition)
    # new_prompt = Prompt.ask(
    #     "Enter a new prompt if you want to change it", default=vima_instance.prompt
    # )
    # evaluation.environment.update_prompt(new_prompt)
    # Run the instance in the environment
    typer.confirm("Press enter to start")

    with torch.inference_mode():
        evaluation.run_vima_instance(vima_instance)

    console.print(evaluation._metric.compute())  # noqa: SLF001
    typer.confirm("Press enter to exit")


if __name__ == "__main__":
    app()
