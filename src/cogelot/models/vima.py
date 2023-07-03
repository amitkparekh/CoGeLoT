from typing import TYPE_CHECKING, cast, get_args

import torch
from lightning import pytorch as pl
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy

from cogelot.modules.preprocessors.their_instance_batcher import (
    TheirInstanceBatcher,
    collate_target_action_tokens,
)
from cogelot.structures.model import ModelInstance, PreprocessedInstance
from cogelot.structures.vima import PoseActionType
from vima.nn.action_decoder.dists import MultiCategorical
from vima.policy import VIMAPolicy


if TYPE_CHECKING:
    from jaxtyping import Float


class VIMALightningModule(pl.LightningModule):
    """Lighting module for a VIMA model."""

    ignore_target_index: int = -100

    def __init__(self, *, vima_policy: VIMAPolicy) -> None:
        super().__init__()

        self.vima_policy = vima_policy
        self.instance_batcher = TheirInstanceBatcher(vima_policy)

        self.loss = MeanMetric()
        self.accuracy = MulticlassAccuracy(num_classes=100, ignore_index=self.ignore_target_index)

    def forward(
        self, instances: list[PreprocessedInstance]
    ) -> dict[PoseActionType, MultiCategorical]:
        """Perform the forward on a batch of instances."""
        # Embed and batch the instances
        prepared_batch: ModelInstance = self.instance_batcher(instances)

        # Encode the prompt
        encoded_prompt = self.vima_policy.forward_prepared_prompt(
            prepared_batch.embedded_prompt, prepared_batch.embedded_prompt_mask
        )

        encoded_predicted_actions: Float[torch.Tensor, "num_obs batch dim"] = (
            self.vima_policy.forward(
                obs_token=prepared_batch.embedded_observations,
                action_token=prepared_batch.embedded_actions,
                prompt_token=encoded_prompt,
                prompt_token_mask=prepared_batch.embedded_prompt_mask,
                obs_mask=prepared_batch.embedded_observations_mask,
            )
        )

        predicted_actions_dists: dict[PoseActionType, MultiCategorical] = (
            self.vima_policy.forward_action_decoder(encoded_predicted_actions)
        )

        return predicted_actions_dists

    def training_step(self, batch: list[PreprocessedInstance]) -> torch.Tensor:
        """Perform a training step."""
        predicted_actions = self.forward(batch)
        target_actions = collate_target_action_tokens(batch)

        loss = self._compute_loss(predicted_actions, target_actions)
        return loss

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
                self.accuracy.update(predicted_logits.argmax(-1), target_axis.reshape(-1))

        # Convert loss list to tensor
        stacked_loss = torch.stack(losses)

        # Update the loss tracker for the epoch
        self.loss.update(stacked_loss)

        # Return the average loss for the batch
        average_loss = torch.mean(stacked_loss)
        return average_loss
