import math
from collections.abc import Mapping
from typing import Any, ClassVar, cast

import numpy as np
import polars as pl
import torch
import wandb
from loguru import logger
from polars.exceptions import ColumnNotFoundError
from torch import distributed as dist
from torchmetrics import MeanMetric, Metric, SumMetric

from cogelot.common.system_deps import verify_ffmpeg_is_available
from cogelot.common.wandb import log_table_to_wandb
from cogelot.structures.common import ObservationVideos
from cogelot.structures.vima import (
    Partition,
    Task,
    VIMAInstance,
    VIMAInstanceMetadata,
)


def compute_hesitance(success_tracker_per_step: list[bool]) -> float:
    """Compute the hesitance of an episode from the tracked success statuses.

    Hesitance is calucated as the ratio of the False's that come after the first True.

    If hesitance is ever < 1, then it means that the model succeeded at some point, but decided
    against it.
    """
    # If there are no successes, then the hesitance is nan
    try:
        first_success_index = success_tracker_per_step.index(True)
    except ValueError:
        return float("nan")

    # +1 beacuse we don't want to include the first success
    tracker_after_first_success = success_tracker_per_step[first_success_index + 1 :]

    try:
        hesitance = tracker_after_first_success.count(False) / len(tracker_after_first_success)
    except ZeroDivisionError:
        hesitance = 0

    return hesitance


def compute_flailing(success_tracker_per_step: list[bool]) -> float:
    """Compute how much the robot flailed in an episode from the tracked success statuses.

    Flailing is caluclated from the number of steps AFTER the first success.

    Flailing has two parts: just waving around like nothing, and causing harm to progress. We are
    just tracking the first part, which is how long the robot continues to make steps after
    the first success. The longer the steps, the worst it is.

    f(x) = tanh(ln(x)), therefore f is bounded by [0, 1] for any x > 0. In case the number of steps
    is 0, we just return 0. This function is incredibly strict, as flailing for 2 steps returns a
    value of 0.6.
    """
    # If there are no successes, then the hesitance is nan
    try:
        first_success_index = success_tracker_per_step.index(True)
    except ValueError:
        return float("nan")

    # +1 beacuse we don't want to include the first success
    tracker_after_first_success = success_tracker_per_step[first_success_index + 1 :]

    num_steps_flailing = len(tracker_after_first_success)
    try:
        flail_value = math.tanh(math.log(num_steps_flailing))
    except ValueError:
        flail_value = 0
    return flail_value


class EvaluationEpisodeTracker:
    """Track and compute metrics during online evaluation.

    Frame width and height are from manual inspection of the data during debugging.
    """

    video_columns: ClassVar[list[str]] = ["top_rgb", "front_rgb", "top_segm", "front_segm"]

    def __init__(self, *, save_observations: bool = False, disable_upload: bool = False) -> None:
        self._disable_upload = disable_upload
        self._save_observations = save_observations

        self._schema_overrides = {
            **VIMAInstanceMetadata.polars_schema_override(),
        }

        if self._save_observations:
            verify_ffmpeg_is_available()
            self._schema_overrides.update(ObservationVideos.polars_schema_override())

        self.table: pl.DataFrame = pl.DataFrame(schema_overrides=self._schema_overrides)

    def update(self, *, vima_instance: VIMAInstance) -> None:
        """Add the result of an episode to the metric."""
        new_row: dict[str, Any] = {
            # "flailing_rate": compute_flailing(success_tracker_per_step),
            # "hesitance_rate": compute_hesitance(success_tracker_per_step),
        }
        new_row.update(vima_instance.to_metadata().model_dump())

        if self._save_observations:
            observation_videos = vima_instance.observations.convert_to_videos().as_python()
            new_row.update(observation_videos)

        # Every value needs to be wrapped in a list because dataframes.
        new_row = {k: [v] for k, v in new_row.items()}
        table = pl.DataFrame(new_row, schema_overrides=self._schema_overrides)

        self.table = (
            table
            if self.table.is_empty()
            else pl.concat([self.table, table], how="diagonal_relaxed")
        )

    def compute_table(self) -> wandb.Table:
        """Compute and return the table."""
        # if update has not been called yet, this WILL fail and crash things. That's the point.
        wandb_table = wandb.Table(
            columns=[column for column in self.table.columns if column not in self.video_columns],
            allow_mixed_types=True,
        )

        # We want to split off the videos from the metadata
        metadata_table = self.table.select(
            *[column for column in self.table.columns if column not in self.video_columns]
        )
        # Add all the metadata to the table
        logger.info("Converting metadata to wandb table")
        for row in metadata_table.to_dicts():
            wandb_table.add_data(*list(row.values()))

        try:
            videos_table = self.table.select(*self.video_columns)
        except ColumnNotFoundError:
            return wandb_table

        # Convert each row into a wandb Video
        for column_name in videos_table.columns:
            videos_per_row = [
                wandb.Video(np.array(array, dtype=np.uint8), caption=column_name, fps=1)
                for array in videos_table[column_name].to_list()
            ]
            wandb_table.add_column(column_name, videos_per_row)

        return wandb_table

    def sync(self) -> None:
        """Sync all the tables across processes."""
        # Nothing happens if we're not in distributed mode
        if not dist.is_initialized():
            return

        dist.barrier()
        all_tables: list[None] | list[pl.DataFrame] = [None for _ in range(dist.get_world_size())]
        dist.gather_object(self.table, all_tables if dist.get_rank() == 0 else None, dst=0)
        if dist.get_rank() == 0:
            self.table = pl.concat(cast(list[pl.DataFrame], all_tables), how="diagonal_relaxed")

    def upload_table(self) -> None:
        """Upload the table to wandb."""
        self.sync()
        if self._disable_upload:
            return

        # Only the master process should be uploading the table
        if not dist.is_initialized() or dist.get_rank() == 0:
            wandb_table = self.compute_table()
            logger.info("Uploading episodes table to WandB")
            log_table_to_wandb(name="episodes", table=wandb_table)


class OnlineEvaluationMetrics(Metric):
    """Track and compute metrics for the online evaluation."""

    key_template: ClassVar[str] = "{metric}/L{partition}/Task{task}"

    def __init__(self) -> None:
        super().__init__()
        self.success_rate = {
            partition: {task: MeanMetric() for task in Task} for partition in Partition
        }
        self.steps_taken = {
            partition: {task: MeanMetric() for task in Task} for partition in Partition
        }
        self.tasks_seen = {
            partition: {task: SumMetric() for task in Task} for partition in Partition
        }

    def update(
        self,
        partition: Partition,
        task: Task,
        *,
        success_tracker_per_step: list[bool],
        num_steps_taken: int,
    ) -> None:
        """Update the metric with the result of an episode."""
        logger.debug(f"Updating metric for {partition}/{task}")

        # To be successful means it ends successfully
        self.success_rate[partition][task](int(success_tracker_per_step[-1]))
        self.steps_taken[partition][task](num_steps_taken)
        self.tasks_seen[partition][task](1)

    def compute(self) -> dict[str, torch.Tensor | float]:  # noqa: WPS210
        """Compute the metrics, returning a flattened dict of all metrics.

        For any partition/task, if the number of steps taken is 0, we do not report it at all.
        """
        seen_per_task_per_partition = {
            partition: {
                task: count.compute() for task, count in task_counts.items() if count.update_called
            }
            for partition, task_counts in self.tasks_seen.items()
        }

        computed_tasks_seen = self._compute_rates(
            metrics_per_task_per_partition=seen_per_task_per_partition,
            metric_name="seen",
            seen_per_task_per_partition=seen_per_task_per_partition,
        )
        computed_steps = self._compute_rates(
            metrics_per_task_per_partition=self.steps_taken,
            metric_name="steps",
            seen_per_task_per_partition=seen_per_task_per_partition,
        )
        computed_success_rate = self._compute_rates(
            metrics_per_task_per_partition=self.success_rate,
            metric_name="success",
            seen_per_task_per_partition=seen_per_task_per_partition,
        )

        total_seen = sum(computed_tasks_seen.values())
        total_success = self._compute_overall_success_rate(
            computed_tasks_seen=computed_tasks_seen, computed_success_rate=computed_success_rate
        )

        return {
            "total/seen": total_seen,
            "total/success": total_success,
            **computed_tasks_seen,
            **computed_steps,
            **computed_success_rate,
        }

    def _compute_rates(
        self,
        *,
        metrics_per_task_per_partition: Mapping[Partition, Mapping[Task, Metric | torch.Tensor]],
        metric_name: str,
        seen_per_task_per_partition: dict[Partition, dict[Task, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Compute the various rates for a given metric into a flattened dictionary.

        This involves a huge dictionary comprehension, but it's also the fastest way to do this
        madness.
        """
        computed_metrics = {
            self.key_template.format(
                partition=partition.value, task=str(task.value + 1).zfill(2), metric=metric_name
            ): metric.compute() if isinstance(metric, Metric) else metric
            for partition, metrics_per_task in metrics_per_task_per_partition.items()
            for task, metric in metrics_per_task.items()
            if task in seen_per_task_per_partition[partition]
            and seen_per_task_per_partition[partition][task] > 0
        }
        return computed_metrics

    def _compute_overall_success_rate(
        self,
        computed_tasks_seen: dict[str, torch.Tensor],
        computed_success_rate: dict[str, torch.Tensor],
    ) -> torch.Tensor | float:
        """Compute the overall success rate across all tasks."""
        relevant_successes = [
            computed_success_rate[seen_key.replace("seen", "success")] * seen_value
            for seen_key, seen_value in computed_tasks_seen.items()
        ]
        total_success = sum(relevant_successes) / sum(computed_tasks_seen.values())
        return total_success
