from typing import ClassVar, Self

import torch
from torchmetrics import MeanMetric
from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.wrappers import MultitaskWrapper

from cogelot.structures.vima import PARTITION_PER_LEVEL, TASK_PER_INDEX


class PoseAccuracyMetric(MultitaskWrapper):
    """Accuracy metric for the pose action."""

    @classmethod
    def from_config(
        cls,
        *,
        max_num_pose_position_classes: int,
        max_num_pose_rotation_classes: int,
        ignore_index: int
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


class OnlineEvaluationMetric:
    """Track and compute metrics for the online evaluation."""

    partitions: ClassVar[tuple[int]] = tuple(PARTITION_PER_LEVEL.keys())
    tasks: ClassVar[tuple[int]] = tuple(TASK_PER_INDEX.keys())

    key_template: ClassVar[str] = "L{partition}_Task{task}_{metric}"

    def __init__(self) -> None:
        self.success_rate = {
            partition: {task: MeanMetric() for task in self.tasks} for partition in self.partitions
        }
        self.steps_taken = {
            partition: {task: MeanMetric() for task in self.tasks} for partition in self.partitions
        }

    def update(
        self, partition: int, task: int, *, is_successful: bool, num_steps_taken: int
    ) -> None:
        """Update the metric with the result of an episode."""
        self.success_rate[partition][task](int(is_successful))
        self.steps_taken[partition][task](num_steps_taken)

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute the metrics, returning a flattened dict of all metrics.

        For any partition/task, if the number of steps taken is 0, we do not report it at all.
        """
        steps_per_task_per_partition = {
            partition: {task: metric.compute() for task, metric in task_metrics.items()}
            for partition, task_metrics in self.steps_taken.items()
        }

        # We want to remove any metrics where the step count is equal to 0, to avoid confusion
        computed_steps = {
            self.key_template.format(partition=partition, task=task, metric="steps"): steps
            for partition, steps_per_task in steps_per_task_per_partition.items()
            for task, steps in steps_per_task.items()
            if steps > 0
        }

        computed_success_rate = {
            self.key_template.format(
                partition=partition, task=task, metric="success"
            ): success_rate.compute()
            for partition, success_rate_per_task in self.success_rate.items()
            for task, success_rate in success_rate_per_task.items()
            if steps_per_task_per_partition[partition][task] > 0
        }

        return {**computed_steps, **computed_success_rate}
