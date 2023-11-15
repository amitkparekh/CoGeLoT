from collections.abc import Mapping
from typing import Any, ClassVar, Literal, cast, get_args

import torch
import wandb
from loguru import logger
from torchmetrics import MeanMetric, SumMetric
from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric
from torchmetrics.wrappers import MultitaskWrapper
from torchmetrics.wrappers.abstract import WrapperMetric

from cogelot.nn.loss import PER_AXIS_KEY_TEMPLATE, PerActionPerAxis
from cogelot.structures.vima import (
    AxesPerPoseActionType,
    Partition,
    PoseActionType,
    Task,
    get_task_group_from_task,
)
from vima.nn.action_decoder.dists import MultiCategorical

TrainingSplit = Literal["train", "val", "test"]


def _create_metric_key_per_axis(string_template: str = PER_AXIS_KEY_TEMPLATE) -> list[str]:
    """Create keys for every single axis."""
    return [
        string_template.format(pose_action_type=pose_action_type, axis=axis)
        for pose_action_type, axis_literal in AxesPerPoseActionType.items()
        for axis in get_args(axis_literal)
    ]


def _include_task_in_metric_key(metric_key: str, task: Task) -> str:
    """Include the task in the metric key."""
    return f"task{task.value + 1:02d}_{metric_key}"


class WrappedMultitaskWrapper(WrapperMetric):
    """Wrap the multitask wrapper so we have less type errors.

    Unnecessary but they were annoying me. This fixes that by doing some patching.
    """

    is_differentiable = False

    def __init__(self, metrics: Mapping[str, Metric], *, compute_on_cpu: bool = False) -> None:
        super().__init__(compute_on_cpu=compute_on_cpu)
        self.metrics = MultitaskWrapper(cast(dict[str, Metric | MetricCollection], metrics))
        self.task_metrics = self.metrics.task_metrics

    def compute(self) -> dict[str, Any]:
        """Compute metrics for all tasks."""
        return self.metrics.compute()

    def reset(self) -> None:
        """Reset all underlying metrics."""
        return self.metrics.reset()


class PoseAccuracyPerAxisMetric(WrappedMultitaskWrapper):
    """Accuracy metric for the pose action."""

    def __init__(
        self, max_num_classes: int, ignore_index: int, *, compute_on_cpu: bool = False
    ) -> None:
        metric_keys = _create_metric_key_per_axis()
        metrics = {
            key: MulticlassAccuracy(
                num_classes=max_num_classes,
                ignore_index=ignore_index,
                compute_on_cpu=compute_on_cpu,
            )
            for key in metric_keys
        }
        super().__init__(metrics)

    def update(
        self,
        predictions: dict[PoseActionType, torch.Tensor],
        targets: dict[PoseActionType, torch.Tensor],
    ) -> None:
        """Update the metrics for each axis."""
        if predictions.keys() != targets.keys():
            raise ValueError(
                f"Keys for the predictions ({predictions.keys()}) for not match the keys for the"
                f" targets ({targets.keys()})"
            )

        predictions_per_axis = PerActionPerAxis.from_actions(predictions).to_flattened_dict()
        targets_per_axis = PerActionPerAxis.from_actions(targets).to_flattened_dict()

        self.metrics.update(predictions_per_axis, targets_per_axis)


class PoseAccuracyPerAxisPerTaskMetric(WrappedMultitaskWrapper):
    """Accuracy metric for each axis for each pose, separated by task."""

    def __init__(
        self, max_num_classes: int, ignore_index: int, *, compute_on_cpu: bool = False
    ) -> None:
        metric_keys = [
            _include_task_in_metric_key(key, task)
            for key in _create_metric_key_per_axis()
            for task in Task
        ]
        metrics = {
            key: MulticlassAccuracy(
                num_classes=max_num_classes,
                ignore_index=ignore_index,
                compute_on_cpu=compute_on_cpu,
            )
            for key in metric_keys
        }
        super().__init__(metrics)

    def update(
        self,
        predictions: dict[PoseActionType, torch.Tensor],
        targets: dict[PoseActionType, torch.Tensor],
        *,
        tasks: list[Task],
    ) -> None:
        """Update the metrics for each axis."""
        if predictions.keys() != targets.keys():
            raise ValueError(
                f"Keys for the predictions ({predictions.keys()}) for not match the keys for the"
                f" targets ({targets.keys()})"
            )

        predictions_per_axis = PerActionPerAxis.from_actions(predictions).to_flattened_dict()
        targets_per_axis = PerActionPerAxis.from_actions(targets).to_flattened_dict()

        # Split each tensor within each dictionary into a list of tensors, one for each task. We
        # are splitting across the batch dimension.
        predictions_per_task_per_axis = (
            (task, {key: tensor[idx] for key, tensor in predictions_per_axis.items()})
            for idx, task in enumerate(tasks)
        )
        targets_per_task_per_axis = (
            (task, {key: tensor[idx] for key, tensor in targets_per_axis.items()})
            for idx, task in enumerate(tasks)
        )

        iterator = zip(predictions_per_task_per_axis, targets_per_task_per_axis, strict=True)

        for (task, task_predictions_per_axis), (_, task_target_per_axis) in iterator:
            flattened_predictions_per_axis = {
                _include_task_in_metric_key(key, task): tensor
                for key, tensor in task_predictions_per_axis.items()
            }
            flattened_targets_per_axis = {
                _include_task_in_metric_key(key, task): tensor
                for key, tensor in task_target_per_axis.items()
            }

            for metric_key, tensor in flattened_predictions_per_axis.items():
                self.task_metrics[metric_key](tensor, flattened_targets_per_axis[metric_key])

    def compute(self) -> dict[str, Any]:
        """Only run compute on the metrics that have been updated."""
        return {
            key: cast(MulticlassAccuracy, metric).compute()
            for key, metric in self.task_metrics.items()
            if metric.update_called
        }


class LossPerAxisPerActionMetric(WrappedMultitaskWrapper):
    """Track the loss per axis per action.

    If there are NaN's in the loss, they get ignored. This is because of how NaN's are used to aid
    with masking the loss during reduction.
    """

    def __init__(self, *, compute_on_cpu: bool = False) -> None:
        # Get all of the keys that exist for the loss
        loss_keys = _create_metric_key_per_axis(PER_AXIS_KEY_TEMPLATE)
        # And create the metric for each of them
        metrics = {
            key: MeanMetric(nan_strategy="ignore", compute_on_cpu=compute_on_cpu)
            for key in loss_keys
        }
        super().__init__(metrics)

    def update(self, fine_grained_loss: dict[str, torch.Tensor]) -> None:
        """Update the metric with the result of a batch."""
        if self.task_metrics.keys() != fine_grained_loss.keys():
            raise ValueError(
                f"Keys for loss ({fine_grained_loss.keys()}) do not match keys for metrics "
                f"({self.task_metrics.keys()})"
            )

        for key, loss in fine_grained_loss.items():
            self.task_metrics[key](loss)


class LossPerAxisPerActionPerTaskMetric(WrappedMultitaskWrapper):
    """Track the loss per axis per action per task.

    If there are NaN's in the loss, they get ignored. This is because of how NaN's are used to aid
    with masking the loss during reduction.
    """

    def __init__(self, *, compute_on_cpu: bool = False) -> None:
        # Get all of the keys that exist for the loss
        loss_keys = [
            _include_task_in_metric_key(key, task)
            for key in _create_metric_key_per_axis(PER_AXIS_KEY_TEMPLATE)
            for task in Task
        ]
        # And create the metric for each of them
        metrics = {
            key: MeanMetric(nan_strategy="ignore", compute_on_cpu=compute_on_cpu)
            for key in loss_keys
        }
        super().__init__(metrics)

    def update(self, fine_grained_loss: dict[str, torch.Tensor], *, tasks: list[Task]) -> None:
        """Update the metric with the result of a batch."""
        # Split each tensor within each dictionary into a list of tensors, one for each task.
        loss_per_task_per_axis = (
            (task, {key: tensor[idx] for key, tensor in fine_grained_loss.items()})
            for idx, task in enumerate(tasks)
        )

        for task, loss_per_axis in loss_per_task_per_axis:
            flattened_loss_per_axis = {
                _include_task_in_metric_key(key, task): tensor
                for key, tensor in loss_per_axis.items()
            }

            for metric_key, loss in flattened_loss_per_axis.items():
                self.task_metrics[metric_key](loss)

    def compute(self) -> dict[str, Any]:
        """Only run compute on the metrics that have been updated."""
        return {
            key: cast(MeanMetric, metric).compute()
            for key, metric in self.task_metrics.items()
            if metric.update_called
        }


class TrainingMetrics(torch.nn.Module):
    """Track and compute metrics for the training."""

    def __init__(
        self, max_num_classes: int, ignore_index: int, *, compute_on_cpu: bool = False
    ) -> None:
        super().__init__()
        self.pose_accuracy_per_axis = PoseAccuracyPerAxisMetric(
            max_num_classes=max_num_classes,
            ignore_index=ignore_index,
            compute_on_cpu=compute_on_cpu,
        )
        self.pose_accuracy_per_axis_per_task = PoseAccuracyPerAxisPerTaskMetric(
            max_num_classes=max_num_classes,
            ignore_index=ignore_index,
            compute_on_cpu=compute_on_cpu,
        )
        self.loss_per_axis_per_action = LossPerAxisPerActionMetric(compute_on_cpu=compute_on_cpu)
        self.loss_per_axis_per_action_per_task = LossPerAxisPerActionPerTaskMetric(
            compute_on_cpu=compute_on_cpu
        )
        self.examples_seen = SumMetric(compute_on_cpu=compute_on_cpu)

    @torch.no_grad()
    def update_accuracy(
        self,
        predictions: dict[PoseActionType, MultiCategorical],
        targets: dict[PoseActionType, torch.Tensor],
        *,
        tasks: list[Task],
        split: TrainingSplit,
    ) -> None:
        """Update the accuracy metrics."""
        if split == "train":
            return

        predicted_action_tokens: dict[PoseActionType, torch.Tensor] = {
            pose_action_type: action_distribution.mode()
            for pose_action_type, action_distribution in predictions.items()
        }
        self.pose_accuracy_per_axis.update(predicted_action_tokens, targets)

        if split == "test":
            self.pose_accuracy_per_axis_per_task.update(
                predicted_action_tokens, targets, tasks=tasks
            )

    @torch.no_grad()
    def update_loss(
        self,
        fine_grained_loss: dict[str, torch.Tensor],
        *,
        tasks: list[Task],
        split: TrainingSplit,
    ) -> None:
        """Update the loss metrics."""
        if split == "train":
            return

        self.loss_per_axis_per_action.update(fine_grained_loss)

        if split == "test":
            self.loss_per_axis_per_action_per_task.update(fine_grained_loss, tasks=tasks)

    @torch.no_grad()
    def update_examples_seen(self, batch_size: int) -> None:
        """Update the number of examples seen."""
        self.examples_seen(batch_size)

    @torch.no_grad()
    def compute(self, split: TrainingSplit) -> dict[str, torch.Tensor]:
        """Compute the metrics, returning a flattened dict of all metrics."""
        metrics = {}

        # We only want to return the examples seen metric during training since it doesn't matter
        # during validation and shouldn't change.
        if split == "train":
            metrics["trainer/examples_seen"] = self.examples_seen.compute()

        if split in ("val", "test"):
            pose_accuracy_per_axis = {
                f"{split}_{key}_acc": acc
                for key, acc in self.pose_accuracy_per_axis.compute().items()
            }
            loss_per_axis = {
                f"{split}_{key}_loss": loss
                for key, loss in self.loss_per_axis_per_action.compute().items()
            }
            metrics.update(loss_per_axis)
            metrics.update(pose_accuracy_per_axis)

            metrics[f"{split}_acc"] = torch.mean(
                torch.stack(list(pose_accuracy_per_axis.values()))
            )

        if split == "test":
            pose_accuracy_per_axis_per_task = {
                f"{split}_{key}_acc": acc
                for key, acc in self.pose_accuracy_per_axis_per_task.compute().items()
            }
            loss_per_axis_per_action_per_task = {
                f"{split}_{key}_loss": loss
                for key, loss in self.loss_per_axis_per_action_per_task.compute().items()
            }
            metrics.update(loss_per_axis_per_action_per_task)
            metrics.update(pose_accuracy_per_axis_per_task)

        return metrics

    def reset(self) -> None:
        """Reset all of the metrics."""
        self.pose_accuracy_per_axis.reset()
        self.pose_accuracy_per_axis_per_task.reset()
        self.loss_per_axis_per_action.reset()
        self.loss_per_axis_per_action_per_task.reset()


class EvaluationMetrics:
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
        self.steps_taken = {
            partition: {task: MeanMetric() for task in Task} for partition in Partition
        }
        self.tasks_seen = {
            partition: {task: SumMetric() for task in Task} for partition in Partition
        }

        self.episode_success_table = wandb.Table(columns=self.columns)

    def update(
        self, partition: Partition, task: Task, *, is_successful: bool, num_steps_taken: int
    ) -> None:
        """Update the metric with the result of an episode."""
        logger.debug(f"Updating metric for {partition}/{task}")

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

        return {**computed_tasks_seen, **computed_steps, **computed_success_rate}
