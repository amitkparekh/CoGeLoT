from typing import Literal, get_args

import torch
from einops import rearrange
from torchmetrics import (
    ClasswiseWrapper,
    MeanMetric,
    MultioutputWrapper,
    SumMetric,
)
from torchmetrics.classification import MulticlassAccuracy, MulticlassExactMatch

from cogelot.structures.vima import AxesPerPoseActionType, Task

METRIC_LABELS = tuple(
    f"{pose}_{axis}"
    for pose, axis_literal in sorted(AxesPerPoseActionType.items())
    for axis in get_args(axis_literal)
)


class OfflineMetrics(torch.nn.Module):
    """Common metrics across both splits for tracking offline performance."""

    def __init__(
        self,
        split_name_prefix: Literal["train", "val"],
        num_axes: int,
        max_num_classes: int,
        ignore_index: int,
    ) -> None:
        super().__init__()
        self.split_name = split_name_prefix
        self.num_axes = num_axes

        self.accuracy = MulticlassExactMatch(
            num_classes=max_num_classes, ignore_index=ignore_index
        )

        self.loss_per_axis = ClasswiseWrapper(
            MultioutputWrapper(MeanMetric(), num_outputs=self.num_axes),
            labels=list(METRIC_LABELS),
            prefix=f"{self.split_name}/",
            postfix="/loss",
        )
        self.accuracy_per_axis = ClasswiseWrapper(
            MulticlassAccuracy(
                num_classes=max_num_classes,
                multidim_average="samplewise",
                ignore_index=ignore_index,
            ),
            labels=list(METRIC_LABELS),
            prefix=f"{self.split_name}/",
            postfix="/acc",
        )

    @torch.no_grad()
    def update(
        self,
        *,
        fine_grained_loss: torch.Tensor,
        predicted_actions: torch.Tensor,
        target_actions: torch.Tensor,
        tasks: list[Task],
    ) -> None:
        """Update the metrics."""
        self.update_loss(fine_grained_loss, tasks=tasks)
        self.update_accuracy(predicted_actions, target_actions, tasks=tasks)

    @torch.no_grad()
    def update_loss(
        self,
        fine_grained_loss: torch.Tensor,
        *,
        tasks: list[Task],  # noqa: ARG002
    ) -> None:
        """Update the loss metric from the fine-grained loss."""
        self.loss_per_axis.update(fine_grained_loss)

    @torch.no_grad()
    def update_accuracy(
        self,
        predicted_actions: torch.Tensor,
        target_actions: torch.Tensor,
        *,
        tasks: list[Task],  # noqa: ARG002
    ) -> None:
        """Update the accuracy metric from the predicted and target actions."""
        predicted_actions = rearrange(predicted_actions, "pose bsz obs dim -> pose dim bsz obs")
        self.accuracy_per_axis.update(predicted_actions, target_actions)

        self.accuracy.update(
            preds=rearrange(predicted_actions, "pose dim bsz obs -> (pose bsz obs) dim"),
            target=rearrange(target_actions, "pose bsz obs -> (pose bsz obs)"),
        )

    @torch.no_grad()
    def compute(self) -> dict[str, torch.Tensor]:
        """Compute and return the metrics."""
        metrics: dict[str, torch.Tensor] = {}
        metrics[f"{self.split_name}_acc"] = self.accuracy.compute()
        metrics.update(self.loss_per_axis.compute())
        metrics.update(self.accuracy_per_axis.compute())
        return metrics

    def reset(self) -> None:
        """Reset the metrics."""
        self.loss_per_axis.reset()
        self.accuracy_per_axis.reset()
        self.accuracy.reset()


class TrainingMetrics(OfflineMetrics):
    """Metrics for tracking training performance."""

    def __init__(
        self,
        num_axes: int,
        max_num_classes: int,
        ignore_index: int,
        *,
        split_name_prefix: Literal["train", "val"] = "train",
    ) -> None:
        super().__init__(
            split_name_prefix=split_name_prefix,
            num_axes=num_axes,
            max_num_classes=max_num_classes,
            ignore_index=ignore_index,
        )
        self.examples_seen = SumMetric()

    @torch.no_grad()
    def update(
        self,
        *,
        fine_grained_loss: torch.Tensor,
        predicted_actions: torch.Tensor,
        target_actions: torch.Tensor,
        tasks: list[Task],
    ) -> None:
        """Update the metrics."""
        self.update_examples_seen(len(tasks))
        super().update(
            fine_grained_loss=fine_grained_loss,
            predicted_actions=predicted_actions,
            target_actions=target_actions,
            tasks=tasks,
        )

    @torch.no_grad()
    def update_examples_seen(self, batch_size: int) -> None:
        """Update the number of examples seen."""
        self.examples_seen(batch_size)

    @torch.no_grad()
    def compute(self) -> dict[str, torch.Tensor]:
        """Compute and return the metrics."""
        metrics: dict[str, torch.Tensor] = super().compute()
        metrics["examples_seen"] = self.examples_seen.compute()
        return metrics


class ValidationMetrics(OfflineMetrics):
    """Metrics for tracking validation performance."""

    def __init__(
        self,
        num_axes: int,
        max_num_classes: int,
        ignore_index: int,
        *,
        split_name_prefix: Literal["train", "val"] = "val",
    ) -> None:
        super().__init__(
            split_name_prefix=split_name_prefix,
            num_axes=num_axes,
            max_num_classes=max_num_classes,
            ignore_index=ignore_index,
        )
        self.loss_per_axis_per_task = torch.nn.ModuleDict(
            {
                task.name: ClasswiseWrapper(
                    MultioutputWrapper(MeanMetric(), num_outputs=self.num_axes),
                    labels=list(METRIC_LABELS),
                    prefix=f"{self.split_name}/",
                    postfix=f"/task{task.value + 1:02d}/loss",
                )
                for task in Task
            }
        )
        self.accuracy_per_axis_per_task = torch.nn.ModuleDict(
            {
                task.name: ClasswiseWrapper(
                    MulticlassAccuracy(
                        num_classes=max_num_classes,
                        multidim_average="samplewise",
                        ignore_index=ignore_index,
                    ),
                    labels=list(METRIC_LABELS),
                    prefix=f"{self.split_name}/",
                    postfix=f"/task{task.value + 1:02d}/acc",
                )
                for task in Task
            }
        )

    @classmethod
    def without_per_task_metrics(
        cls,
        num_axes: int,
        max_num_classes: int,
        ignore_index: int,
    ) -> TrainingMetrics:
        """Instantiate the metrics without per-task metrics.

        This is basically just the training metrics.
        """
        return TrainingMetrics(
            num_axes=num_axes,
            max_num_classes=max_num_classes,
            ignore_index=ignore_index,
            split_name_prefix="val",
        )

    @torch.no_grad()
    def update_loss(self, fine_grained_loss: torch.Tensor, *, tasks: list[Task]) -> None:
        """Update the loss metrics."""
        super().update_loss(fine_grained_loss, tasks=tasks)

        for task, loss in zip(tasks, fine_grained_loss.unbind(dim=0), strict=True):
            self.loss_per_axis_per_task[task.name].update(loss)

    @torch.no_grad()
    def update_accuracy(
        self, predicted_actions: torch.Tensor, target_actions: torch.Tensor, *, tasks: list[Task]
    ) -> None:
        """Update the accuracy metrics.

        Shapes:
            predicted_actions: (pose, bsz, obs, dim)
            target_actions: (pose, bsz, obs)
        """
        super().update_accuracy(predicted_actions, target_actions, tasks=tasks)

        predicted_actions = rearrange(predicted_actions, "pose bsz obs dim -> pose dim bsz obs")
        for task, predicted, target in zip(
            tasks, predicted_actions.unbind(dim=2), target_actions.unbind(dim=1), strict=True
        ):
            self.accuracy_per_axis_per_task[task.name].update(predicted, target)

    @torch.no_grad()
    def compute(self) -> dict[str, torch.Tensor]:
        """Compute and return the metrics."""
        metrics: dict[str, torch.Tensor] = super().compute()

        for loss_per_task in self.loss_per_axis_per_task.values():
            # If any of the metrics in the MultioutputWrapper have been called, then compute
            if loss_per_task.metric.metrics[0].update_called:
                metrics.update(loss_per_task.compute())

        for accuracy_per_task in self.accuracy_per_axis_per_task.values():
            if accuracy_per_task.metric.update_called:
                metrics.update(accuracy_per_task.compute())

        return metrics

    def reset(self) -> None:
        """Reset the metrics."""
        super().reset()
        for loss_per_task in self.loss_per_axis_per_task.values():
            loss_per_task.reset()
        for accuracy_per_task in self.accuracy_per_axis_per_task.values():
            accuracy_per_task.reset()
