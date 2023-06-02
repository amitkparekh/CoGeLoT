from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from tqdm.rich import RateColumn


VIMA_TRAINING_DATA_DIR = Path("vima_v6/")
OUTPUT_FILE = Path("vima_v6_trajectory_metadata.csv")

progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    TaskProgressColumn(),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    RateColumn(unit="it"),  # pyright: ignore[reportGeneralTypeIssues]
)


def extract_trajectory_metadata_for_instance_dir(instance_dir: Path) -> dict[str, Any] | None:
    """Extract trajectory metadata from the instance dir.

    The schema for the trajectory data was found from just loading the data and seeing what was
    inside. I didn't see a Pydantic model or anything.
    """
    trajectory_file = instance_dir.joinpath("trajectory.pkl")

    if not trajectory_file.exists():
        return None

    trajectory_data = pickle.load(trajectory_file.open("rb"))  # noqa: S301

    task_number = instance_dir.stem
    task_name = instance_dir.parent.stem
    table_data = {
        "task": task_name,
        "id": f"{task_name}_{task_number}",
        "seed": trajectory_data["seed"],
        "end_effector_type": trajectory_data["end_effector_type"],
        "num_objects": trajectory_data["n_objects"],
        "difficulty": trajectory_data["difficulty"],
        "success": trajectory_data["success"],
        "num_steps": trajectory_data["steps"],
        "num_observations": trajectory_data["steps"] + 1,
        "prompt": trajectory_data["prompt"],
    }
    return table_data


def save_trajectory_metadata_to_csv(
    all_trajectory_metadata: list[dict[str, Any]],
    output_file: Path,
) -> None:
    """Save the trajectory metadata to a CSV file."""
    if not output_file.suffix.endswith("csv"):
        msg = "Output file must be a CSV file"
        raise ValueError(msg)

    logger.debug("Converting trajectory metadata to DataFrame")
    trajectory_metadata_df = pd.DataFrame(all_trajectory_metadata)

    logger.debug("Saving trajectory metadata to CSV")
    trajectory_metadata_df.to_csv(output_file, index=False)


def extract_trajectory_metadata(vima_training_data_dir: Path, *, output_file: Path) -> None:
    """Extract trajectory metadata from the VIMA training data."""
    logger.debug("Getting all the possible instance directories")
    all_instance_dirs = [
        instance_dir
        for task_dir in vima_training_data_dir.iterdir()
        for instance_dir in task_dir.iterdir()
    ]

    logger.info("Extracting trajectory metadata")
    extraction_progress_task = progress.add_task(
        "Extracting trajectory metadata",
        total=len(all_instance_dirs),
    )
    with progress:
        all_trajectory_metadata: list[dict[str, Any]] = []

        for instance_dir in all_instance_dirs:
            trajectory_data = extract_trajectory_metadata_for_instance_dir(instance_dir)

            if trajectory_data is not None:
                all_trajectory_metadata.append(trajectory_data)

            progress.advance(extraction_progress_task)

        logger.debug("Finished extracting trajectory metadata")

    logger.info(
        (
            f"{len(all_trajectory_metadata)} trajectory metadata extracted."
            f" {len(all_instance_dirs) - len(all_trajectory_metadata)} were not valid"
            " trajectories."
        ),
    )

    # Save the data
    logger.info("Saving trajectory metadata to CSV")
    save_trajectory_metadata_to_csv(all_trajectory_metadata, output_file)

    logger.info("Done!")


if __name__ == "__main__":
    extract_trajectory_metadata(VIMA_TRAINING_DATA_DIR, output_file=OUTPUT_FILE)
