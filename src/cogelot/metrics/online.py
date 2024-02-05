import math
from collections.abc import Mapping
from typing import ClassVar, cast

import polars as pl
import torch
import wandb
from einops import rearrange
from loguru import logger
from matplotlib import pyplot as plt
from pydantic import BaseModel, ConfigDict
from torch import distributed as dist
from torchmetrics import MeanMetric, Metric, SumMetric

from cogelot.common.system_deps import verify_ffmpeg_is_available
from cogelot.common.wandb import log_table_to_wandb
from cogelot.metrics.online_tasks import parse_base_task
from cogelot.structures.common import ImageType, Observation, View
from cogelot.structures.vima import (
    EndEffector,
    Partition,
    Task,
    get_task_group_from_task,
)
from vima_bench.tasks.task_suite.base import BaseTask

# Create a tensor of colors for the segmentation masks
COLOR_MAP: torch.Tensor = (torch.tensor(plt.cm.tab20(range(20)))[:, :-1] * 255).to(torch.uint8)  # pyright: ignore[reportGeneralTypeIssues,reportAttributeAccessIssue] # noqa: WPS221


def _is_zero_rank() -> bool:
    """Return true if currently zero-rank."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


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


class ObservationVideos(BaseModel):
    """Observation videos for a single episode."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    front_rgb: torch.Tensor
    front_segm: torch.Tensor
    top_rgb: torch.Tensor
    top_segm: torch.Tensor


def extract_multiple_videos_from_observations(
    observations: list[Observation],
) -> ObservationVideos:
    """Extract multiple videos from a list of observations."""
    rgb_front_frames = []
    segm_front_frames = []
    rgb_top_frames = []
    segm_top_frames = []

    for observation in observations:
        obs = observation.to_image_per_type_per_view()
        rgb_front_frames.append(obs[View.front][ImageType.rgb])
        segm_front_frames.append(obs[View.front][ImageType.segmentation])
        rgb_top_frames.append(obs[View.top][ImageType.rgb])
        segm_top_frames.append(obs[View.top][ImageType.segmentation])

    front_segmentation = torch.stack(segm_front_frames, dim=0).long()
    top_segmentation = torch.stack(segm_top_frames, dim=0).long()
    colored_front_segmentation = COLOR_MAP[front_segmentation]
    colored_top_segmentation = COLOR_MAP[top_segmentation]

    return ObservationVideos(
        front_rgb=rearrange(rgb_front_frames, "t c h w -> t c h w"),
        top_rgb=rearrange(rgb_top_frames, "t c h w -> t c h w"),
        front_segm=rearrange(colored_front_segmentation, "t h w c -> t c h w"),
        top_segm=rearrange(colored_top_segmentation, "t h w c -> t c h w"),
    )


class EvaluationEpisodeTracker:
    """Track and compute metrics during online evaluation.

    Frame width and height are from manual inspection of the data during debugging.
    """

    def __init__(self, *, save_observations: bool = False) -> None:
        self._save_observations = save_observations
        if self._save_observations:
            verify_ffmpeg_is_available()

        self._schema_overrides = {"success_per_step": pl.List(pl.Boolean)}
        self.table: pl.DataFrame

    def update(
        self,
        *,
        partition: Partition,
        task: Task,
        success_tracker_per_step: list[bool],
        end_effector: EndEffector,
        prompt: str,
        observations: list[Observation],
        environment_task: BaseTask,
        env_seed: int,
        task_seed: int,
        # actions,
    ) -> None:
        """Add the result of an episode to the metric."""
        parsed_environment = parse_base_task(environment_task)
        new_row = {
            "partition": partition.name,
            "task": task.name,
            "task_group": get_task_group_from_task(task).name,
            "end_effector": end_effector,
            "prompt": prompt,
            "env_seed": env_seed,
            "task_seed": task_seed,
            "steps_taken": len(success_tracker_per_step),
            "is_successful_at_end": success_tracker_per_step[-1],
            "success_per_step": success_tracker_per_step,
            "flailing_rate": compute_flailing(success_tracker_per_step),
            "hesitance_rate": compute_hesitance(success_tracker_per_step),
            **parsed_environment["task_meta"],
            "placeholders": parsed_environment["placeholders"],
        }

        if self._save_observations:
            observation_videos = extract_multiple_videos_from_observations(observations)
            new_row["top_rgb"] = observation_videos.top_rgb.numpy()
            new_row["front_rgb"] = observation_videos.front_rgb.numpy()
            new_row["top_segm"] = observation_videos.top_segm.numpy()
            new_row["front_segm"] = observation_videos.front_segm.numpy()

        # Every value needs to be wrapped in a list because dataframes.
        new_row = {k: [v] for k, v in new_row.items()}
        table = pl.DataFrame(new_row, schema_overrides=self._schema_overrides)

        try:
            self.table = pl.concat([self.table, table], how="diagonal_relaxed")
        except AttributeError:
            self.table = table

    def compute_table(self) -> wandb.Table:
        """Compute and return the table."""
        # if update has not been called yet, this WILL fail and crash things. That's the point.
        wandb_table = wandb.Table(columns=self.table.columns, allow_mixed_types=True)
        for row in self.table.to_dicts():
            wandb_table.add_data(*list(row.values()))
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
        if _is_zero_rank():
            wandb_table = self.compute_table()
            log_table_to_wandb(name="episodes", table=wandb_table)


class OnlineEvaluationMetrics:
    """Track and compute metrics for the online evaluation."""

    key_template: ClassVar[str] = "{metric}/L{partition}/Task{task}"

    def __init__(self) -> None:
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
            partition: {task: count.compute() for task, count in task_counts.items()}
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
            if seen_per_task_per_partition[partition][task] > 0
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
