import math
from collections.abc import Mapping
from typing import ClassVar

import torch
import wandb
from loguru import logger
from torchmetrics import MeanMetric, Metric, SumMetric

from cogelot.structures.vima import Partition, Task, get_task_group_from_task


def compute_hesitance(success_tracker_per_step: list[bool]) -> float:
    """Compute the hesitance of an episode from the tracked success statuses.

    Hesitance is calucated as the ratio of the False's that come after the first True.

    If hesitance is ever < 1, then it means that the model succeeded at some point, but decided
    against it.
    """
    first_success_index = success_tracker_per_step.index(True)
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
    first_success_index = success_tracker_per_step.index(True)
    # +1 beacuse we don't want to include the first success
    tracker_after_first_success = success_tracker_per_step[first_success_index + 1 :]

    num_steps_flailing = len(tracker_after_first_success)
    try:
        flail_value = math.tanh(math.log(num_steps_flailing))
    except ValueError:
        flail_value = 0
    return flail_value


class OnlineEvaluationMetrics:
    """Track and compute metrics for the online evaluation."""

    key_template: ClassVar[str] = "{metric}/L{partition}/Task{task}"

    columns: ClassVar[list[str]] = [
        "partition",
        "partition_index",
        "task",
        "task_index",
        "task_group",
        "task_group_index",
        "is_successful",
        "num_steps",
    ]

    def __init__(self) -> None:
        self.success_rate = {
            partition: {task: MeanMetric() for task in Task} for partition in Partition
        }
        self.hesitance_rate = {
            partition: {task: MeanMetric() for task in Task} for partition in Partition
        }
        self.flailing_rate = {
            partition: {task: MeanMetric() for task in Task} for partition in Partition
        }
        self.steps_taken = {
            partition: {task: MeanMetric() for task in Task} for partition in Partition
        }
        self.tasks_seen = {
            partition: {task: SumMetric() for task in Task} for partition in Partition
        }

        self.episode_success_table = wandb.Table(columns=self.columns)

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

        is_successful = success_tracker_per_step[-1]

        # To be successful means it ends successfully
        self.success_rate[partition][task](int(success_tracker_per_step[-1]))
        self.hesitance_rate[partition][task](compute_hesitance(success_tracker_per_step))
        self.flailing_rate[partition][task](compute_flailing(success_tracker_per_step))
        self.steps_taken[partition][task](num_steps_taken)
        self.tasks_seen[partition][task](1)

        task_group = get_task_group_from_task(task)

        self.episode_success_table.add_data(
            partition.name,
            partition.value,
            task.name,
            task.value + 1,
            task_group.name,
            task_group.value,
            1 if is_successful else 0,
            num_steps_taken,
        )

    def compute(self) -> dict[str, torch.Tensor]:
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
        computed_hesitant_rate = self._compute_rates(
            metrics_per_task_per_partition=self.hesitance_rate,
            metric_name="hesitant",
            seen_per_task_per_partition=seen_per_task_per_partition,
        )
        computed_flailing_rate = self._compute_rates(
            metrics_per_task_per_partition=self.flailing_rate,
            metric_name="flailing",
            seen_per_task_per_partition=seen_per_task_per_partition,
        )

        return {
            **computed_tasks_seen,
            **computed_steps,
            **computed_success_rate,
            **computed_hesitant_rate,
            **computed_flailing_rate,
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
