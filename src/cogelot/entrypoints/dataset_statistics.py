from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from rich.pretty import pprint as rich_print

from cogelot.structures.vima import Task, VIMAInstanceMetadata


def convert_metadata_to_polars(filepath: Path) -> pl.DataFrame:
    """Convert the metadata to polars."""
    logger.info("Loading metadata")
    metadata_df = pl.scan_ndjson(
        filepath, schema=VIMAInstanceMetadata.polars_schema_override()
    ).collect()
    return metadata_df


def get_count_hist_per_task(
    metadata_df: pl.DataFrame, *, column_name: str
) -> list[dict[str, Any]]:
    """Get the counts of occurances for a column per task."""
    logger.info("Getting data per task")
    counts_per_task = (
        metadata_df.group_by("task").agg(pl.col(column_name)).sort("task")
    ).to_dicts()

    logger.info("Cleaning stats")
    cleaned_stats = []
    for stats in counts_per_task:
        count_per_bin = np.bincount(stats[column_name]).tolist()
        counter_dict = dict(zip(range(len(count_per_bin)), count_per_bin, strict=True))

        cleaned_stats.append(
            {
                "task": Task(stats["task"]),
                "count_per_bin": counter_dict,
            }
        )

    rich_print(cleaned_stats, expand_all=True)

    return cleaned_stats


if __name__ == "__main__":
    metadata_df = convert_metadata_to_polars(Path("storage/data/dataset_metadata.jsonl"))
    get_count_hist_per_task(metadata_df, column_name="total_steps")
    get_count_hist_per_task(metadata_df, column_name="num_objects")
