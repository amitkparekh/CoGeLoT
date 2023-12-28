from typing import ClassVar

import torch
import wandb
from loguru import logger
from torchmetrics import MeanMetric, SumMetric

from cogelot.structures.vima import Partition, Task, get_task_group_from_task


class SuccessWobbleMetric(MeanMetric):
    """Track the robot confidence after the first success.

    High value is good.
    """

    def update(self, success_tracker_per_step: list[bool]) -> None:  # type: ignore[override]
        """Update the metric with the result of an episode."""
        first_success_index = success_tracker_per_step.index(True)
        confidence_after_first_success = success_tracker_per_step[first_success_index:]

        average_after_first_success = confidence_after_first_success.count(True) / len(
            confidence_after_first_success
        )
        super().update(average_after_first_success)


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
        self.success_wobble_rate = {
            partition: {task: SuccessWobbleMetric() for task in Task} for partition in Partition
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

        # To be successful means it ends successfully
        is_successful = success_tracker_per_step[-1]

        self.success_wobble_rate[partition][task](success_tracker_per_step)
        self.success_rate[partition][task](int(is_successful))
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

        # We want to remove any metrics where no tasks of that type have been seen
        computed_tasks_seen = {
            self.key_template.format(
                partition=partition.value, task=str(task.value + 1).zfill(2), metric="seen"
            ): count
            for partition, task_counts in seen_per_task_per_partition.items()
            for task, count in task_counts.items()
            if count > 0
        }

        computed_steps = {
            self.key_template.format(
                partition=partition.value, task=str(task.value + 1).zfill(2), metric="steps"
            ): steps.compute()
            for partition, steps_per_task in self.steps_taken.items()
            for task, steps in steps_per_task.items()
            if seen_per_task_per_partition[partition][task] > 0
        }

        computed_success_rate = {
            self.key_template.format(
                partition=partition.value, task=str(task.value + 1).zfill(2), metric="success"
            ): success_rate.compute()
            for partition, success_rate_per_task in self.success_rate.items()
            for task, success_rate in success_rate_per_task.items()
            if seen_per_task_per_partition[partition][task] > 0
        }

        computed_success_wobble_rate = {
            self.key_template.format(
                partition=partition.value,
                task=str(task.value + 1).zfill(2),
                metric="success_wobble",
            ): success_wobble_rate.compute()
            for partition, success_wobble_rate_per_task in self.success_wobble_rate.items()
            for task, success_wobble_rate in success_wobble_rate_per_task.items()
            if seen_per_task_per_partition[partition][task] > 0
        }

        return {
            **computed_tasks_seen,
            **computed_steps,
            **computed_success_rate,
            **computed_success_wobble_rate,
        }
