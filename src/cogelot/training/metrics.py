from typing import ClassVar, Self

import torch
from loguru import logger
from torchmetrics import MeanMetric, SumMetric
from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.wrappers import MultitaskWrapper

from cogelot.structures.vima import Partition, Task


class PoseAccuracyMetric(MultitaskWrapper):
    """Accuracy metric for the pose action."""

    @classmethod
    def from_config(
        cls,
        *,
        max_num_pose_position_classes: int,
        max_num_pose_rotation_classes: int,
        ignore_index: int,
    ) -> Self:
        """Create the pose accuracy metric from some hyperparams."""
        return cls(
            {
                "pose0_position": MulticlassAccuracy(
                    num_classes=max_num_pose_position_classes, ignore_index=ignore_index
                ),
                "pose1_position": MulticlassAccuracy(
                    num_classes=max_num_pose_position_classes, ignore_index=ignore_index
                ),
                "pose0_rotation": MulticlassAccuracy(
                    num_classes=max_num_pose_rotation_classes, ignore_index=ignore_index
                ),
                "pose1_rotation": MulticlassAccuracy(
                    num_classes=max_num_pose_rotation_classes, ignore_index=ignore_index
                ),
            }
        )


class EvaluationMetrics:
    """Track and compute metrics for the online evaluation."""

    key_template: ClassVar[str] = "L{partition}_Task{task}_{metric}"

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
        self, partition: Partition, task: Task, *, is_successful: bool, num_steps_taken: int
    ) -> None:
        """Update the metric with the result of an episode."""
        logger.debug(f"Updating metric for Partition {partition}/Task {task}")

        self.success_rate[partition][task](int(is_successful))
        self.steps_taken[partition][task](num_steps_taken)
        self.tasks_seen[partition][task](1)

    def compute_current(self, partition: Partition, task: Task) -> dict[str, torch.Tensor]:
        """Compute metrics for just the provided partition and task."""
        return {
            self.key_template.format(
                partition=partition.value, task=task.value, metric="seen"
            ): self.tasks_seen[partition][task].compute(),
            self.key_template.format(
                partition=partition.value, task=task.value, metric="steps"
            ): self.steps_taken[partition][task].compute(),
            self.key_template.format(
                partition=partition.value, task=task.value, metric="success"
            ): self.success_rate[partition][task].compute(),
        }

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
                partition=partition.value, task=task.value, metric="seen"
            ): count
            for partition, task_counts in seen_per_task_per_partition.items()
            for task, count in task_counts.items()
            if count > 0
        }

        computed_steps = {
            self.key_template.format(
                partition=partition.value, task=task.value, metric="steps"
            ): steps.compute()
            for partition, steps_per_task in self.steps_taken.items()
            for task, steps in steps_per_task.items()
            if seen_per_task_per_partition[partition][task] > 0
        }

        computed_success_rate = {
            self.key_template.format(
                partition=partition.value, task=task.value, metric="success"
            ): success_rate.compute()
            for partition, success_rate_per_task in self.success_rate.items()
            for task, success_rate in success_rate_per_task.items()
            if seen_per_task_per_partition[partition][task] > 0
        }

        return {**computed_tasks_seen, **computed_steps, **computed_success_rate}
