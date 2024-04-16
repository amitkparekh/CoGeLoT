from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import typer
from loguru import logger
from rich.console import Console

from cogelot.common.settings import Settings
from cogelot.entrypoints.preprocess_instances import load_parsed_datasets_for_each_task

if TYPE_CHECKING:
    from cogelot.structures.vima import Task

console = Console()
settings = Settings()


def determine_evaluation_starting_seed(
    parsed_hf_dataset_dir: Annotated[
        Path, typer.Argument(help="Where to get the parsed HF datasets (for each task)")
    ] = settings.parsed_hf_dataset_dir,
) -> None:
    """Iterate the entire dataset to find the largest seed that's been seen."""
    # Track the maximum seed here
    max_seed_per_task: dict[Task, dict[Literal["train", "valid"], int]] = {}

    # Load all the parsed datasets
    parsed_datasets_iterator = load_parsed_datasets_for_each_task(parsed_hf_dataset_dir)

    for task, dataset in parsed_datasets_iterator:
        # Get the maximum seed for each split
        max_seed_per_task[task] = {
            "train": max(dataset["train"]["generation_seed"]),
            "valid": max(dataset["valid"]["generation_seed"]),
        }
        logger.info(f"{task} has max seed: {max_seed_per_task[task]}")

    console.print(max_seed_per_task)

    # Find the maximum seed across all tasks
    max_seed = max(max(max_seed_per_task[task].values()) for task in max_seed_per_task)

    logger.info(f"The maximum seed across all tasks is: {max_seed}")


if __name__ == "__main__":
    determine_evaluation_starting_seed()
