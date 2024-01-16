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
        predicted_actions = rearrange(predicted_actions, "pose bsz obs dim -> pose dim bsz obs")
        self.accuracy_per_axis.update(predicted_actions, target_actions)
        self.accuracy.update(
            preds=rearrange(predicted_actions, "pose dim bsz obs -> (pose bsz obs) dim"),
            target=rearrange(target_actions, "pose bsz obs -> (pose bsz obs)"),
        )

        self.loss_per_axis.update(fine_grained_loss)

        if self.split_name == "train":
            self.examples_seen.update(len(tasks))

    @torch.no_grad()
    def compute(self) -> dict[str, torch.Tensor]:
        """Compute and return the metrics."""
        metrics: dict[str, torch.Tensor] = {}
        metrics[f"{self.split_name}_acc"] = self.accuracy.compute()
        metrics.update(self.loss_per_axis.compute())
        metrics.update(self.accuracy_per_axis.compute())

        if self.split_name == "train":
            metrics["examples_seen"] = self.examples_seen.compute()

        return metrics

    def reset(self) -> None:
        """Reset the metrics."""
        self.loss_per_axis.reset()
        self.accuracy_per_axis.reset()
        self.accuracy.reset()
