import math
from collections.abc import Mapping
from typing import ClassVar

import torch
import wandb
from einops import rearrange
from loguru import logger
from matplotlib import pyplot as plt
from pydantic import BaseModel, ConfigDict
from torchmetrics import MeanMetric, Metric, SumMetric

from cogelot.common.system_deps import verify_ffmpeg_is_available
from cogelot.structures.common import ImageType, Observation, View
from cogelot.structures.vima import (
    EndEffector,
    Partition,
    Task,
    get_task_group_from_task,
)

# Create a tensor of colors for the segmentation masks
COLOR_MAP: torch.Tensor = (plt.cm.tab20(torch.arange(20))[:, :-1] * 255).to(torch.uint8)  # pyright: ignore[reportGeneralTypeIssues] # noqa: WPS221


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

    def __init__(self) -> None:
        verify_ffmpeg_is_available()

        self.wandb_table = wandb.Table(
            columns=[
                "partition",
                "task",
                "task_group",
                "end_effector",
                "prompt",
                "seed",
                "steps_taken",
                "is_successful_at_end",
                # We want to be able to track whether or not the agent completed the task at all
                # timestep.
                "success_per_step",
                "flailing_rate",
                "hesitance_rate",
                # Observations need to be a numpy array with shape: (time, channels, width, height)
                "top_rgb",
                "front_rgb",
                "top_segm",
                "front_segm",
                # Each action has 14 DOF/axes to predict
                # "actions": pl.List(pl.Array(pl.Float32, width=14)),
            ]
        )

    def update(
        self,
        *,
        partition: Partition,
        task: Task,
        seed: int,
        success_tracker_per_step: list[bool],
        end_effector: EndEffector,
        prompt: str,
        observations: list[Observation],
        # actions,
    ) -> None:
        """Add the result of an episode to the metric."""
        observation_videos = extract_multiple_videos_from_observations(observations)

        self.wandb_table.add_data(
            partition.name,
            task.name,
            get_task_group_from_task(task).name,
            end_effector,
            prompt,
            seed,
            len(success_tracker_per_step),
            success_tracker_per_step[-1],
            success_tracker_per_step,
            compute_flailing(success_tracker_per_step),
            compute_hesitance(success_tracker_per_step),
            wandb.Video(observation_videos.top_rgb.numpy(), caption="Top RGB", fps=1),
            wandb.Video(observation_videos.front_rgb.numpy(), caption="Front RGB", fps=1),
            wandb.Video(observation_videos.top_segm.numpy(), caption="Top Segmentation", fps=1),
            wandb.Video(
                observation_videos.front_segm.numpy(), caption="Front Segmentation", fps=1
            ),
        )


class OnlineEvaluationMetrics:
    """Track and compute metrics for the online evaluation."""

    key_template: ClassVar[str] = "{metric}/L{partition}/Task{task}"

    def __init__(self) -> None:
        self.success_rate = {
            partition: {task: MeanMetric() for task in Task} for partition in Partition
        }
        self.hesitance_rate = {
            partition: {task: MeanMetric(nan_strategy="ignore") for task in Task}
            for partition in Partition
        }
        self.flailing_rate = {
            partition: {task: MeanMetric(nan_strategy="ignore") for task in Task}
            for partition in Partition
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
        self.hesitance_rate[partition][task](compute_hesitance(success_tracker_per_step))
        self.flailing_rate[partition][task](compute_flailing(success_tracker_per_step))
        self.steps_taken[partition][task](num_steps_taken)
        self.tasks_seen[partition][task](1)

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
