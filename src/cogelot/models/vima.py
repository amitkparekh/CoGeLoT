from collections.abc import Callable, Iterator
from functools import partial
from typing import Any, LiteralString, cast, get_args

import torch
from lightning import pytorch as pl

from cogelot.modules.policy import Policy
from cogelot.modules.preprocessors.their_instance_batcher import (
    TheirInstanceBatcher,
    collate_target_action_tokens,
)
from cogelot.structures.model import ModelInstance, PreprocessedInstance
from cogelot.structures.vima import PoseActionType
from cogelot.training.metrics import create_pose_accuracy_metric
from vima.nn.action_decoder.dists import MultiCategorical


OptimizerPartialFn = Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer]
LRSchedulerPartialFn = Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]

_default_optimizer = torch.optim.Adam
_default_lr_scheduler = partial(torch.optim.lr_scheduler.ConstantLR, factor=1)


class VIMALightningModule(pl.LightningModule):
    """Lighting module for a VIMA model."""

    ignore_target_index: int = -100

    def __init__(
        self,
        *,
        policy: Policy,
        optimizer_partial_fn: OptimizerPartialFn = _default_optimizer,
        lr_scheduler_partial_fn: LRSchedulerPartialFn = _default_lr_scheduler,
    ) -> None:
        super().__init__()

        self.policy = policy
        self.instance_batcher = TheirInstanceBatcher(self.policy)

        self._optimizer_partial_fn = optimizer_partial_fn
        self._lr_scheduler_partial_fn = lr_scheduler_partial_fn

        self._accuracy = create_pose_accuracy_metric(
            max_num_pose_position_classes=max(
                policy.n_discrete_x_bins, policy.n_discrete_y_bins, policy.n_discrete_z_bins
            ),
            max_num_pose_rotation_classes=policy.n_discrete_rot_bins,
            ignore_index=self.ignore_target_index,
        )

    def forward(
        self, instances: list[PreprocessedInstance]
    ) -> dict[PoseActionType, MultiCategorical]:
        """Perform the forward on a batch of instances."""
        # Embed and batch the instances
        prepared_batch: ModelInstance = self.instance_batcher(instances)

        # Encode the prompt
        encoded_prompt = self.policy.encode_prompt(
            prepared_batch.embedded_prompt, prepared_batch.embedded_prompt_mask
        )

        encoded_predicted_actions = self.policy.predict_action_token(
            encoded_prompt=encoded_prompt,
            encoded_prompt_mask=prepared_batch.embedded_prompt_mask,
            embedded_observations=prepared_batch.embedded_observations,
            embedded_observations_mask=prepared_batch.embedded_observations_mask,
            embedded_actions=prepared_batch.embedded_actions,
        )

        predicted_actions_dists = self.policy.decode_action_token(encoded_predicted_actions)

        return predicted_actions_dists

    def training_step(
        self, batch: list[PreprocessedInstance], batch_idx: int  # noqa: ARG002
    ) -> torch.Tensor:
        """Perform a training step."""
        predicted_actions = self.forward(batch)
        target_actions = collate_target_action_tokens(batch)

        loss = self._compute_loss(predicted_actions, target_actions)

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_acc", self._accuracy, prog_bar=True, logger=True)

        return loss

    def validation_step(
        self, batch: list[PreprocessedInstance], batch_idx: int  # noqa: ARG002
    ) -> torch.Tensor:
        """Perform a validation step (identical to training step)."""
        predicted_actions = self.forward(batch)
        target_actions = collate_target_action_tokens(batch)

        loss = self._compute_loss(predicted_actions, target_actions)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", self._accuracy, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self) -> Any:
        """Configure the optimizer and scheduler."""
        optimizer = self._optimizer_partial_fn(self.parameters())
        scheduler = self._lr_scheduler_partial_fn(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def _compute_loss(
        self,
        predicted_actions: dict[PoseActionType, MultiCategorical],
        target_actions: dict[PoseActionType, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the loss across all the poses."""
        losses = []

        for pose_action_type in get_args(PoseActionType):
            target_for_pose = target_actions[pose_action_type]
            predicted_dist_for_pose = predicted_actions[pose_action_type]

            # We need to split the targets by the number of axes and zip with the various
            # categorical distributions.
            iterator = zip(
                target_for_pose.split(1, dim=-1), predicted_dist_for_pose.dists, strict=True
            )

            # Then we get the loss per axis and add to the list
            for target_axis, predicted_dist_axis in iterator:
                predicted_logits: torch.Tensor = cast(torch.Tensor, predicted_dist_axis.logits)
                predicted_logits = predicted_logits.reshape(-1, predicted_logits.shape[-1])
                losses.append(
                    torch.nn.functional.cross_entropy(
                        predicted_logits,
                        target_axis.reshape(-1),
                        ignore_index=self.ignore_target_index,
                    )
                )

        # Convert loss list to tensor
        stacked_loss = torch.stack(losses)
        # Return the average loss for the batch
        average_loss = torch.mean(stacked_loss)
        return average_loss

    def _update_accuracy(
        self,
        predicted_actions: dict[LiteralString, MultiCategorical],
        target_actions: dict[LiteralString, torch.Tensor],
    ) -> None:
        """Update the accuracy metric for all the pose action types."""
        predicted_action_tokens = {
            pose_action_type: action_distribution.mode()
            for pose_action_type, action_distribution in predicted_actions.items()
        }
        self._accuracy.update(predicted_action_tokens, target_actions)
