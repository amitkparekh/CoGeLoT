from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any, Literal, Self, cast

import torch
from lightning import pytorch as pl
from torchmetrics import SumMetric

from cogelot.common.wandb import download_model_from_wandb
from cogelot.modules.metrics import LossPerAxisPerActionMetric, PoseAccuracyMetric
from cogelot.modules.policy import Policy
from cogelot.nn.loss import compute_fine_grained_loss, reduce_fine_grained_loss
from cogelot.structures.model import ModelInstance, PreprocessedBatch
from cogelot.structures.vima import PoseActionType
from vima.nn.action_decoder.dists import MultiCategorical


OptimizerPartialFn = Callable[[Iterator[torch.nn.Parameter]], torch.optim.Optimizer]
LRSchedulerPartialFn = Callable[..., torch.optim.lr_scheduler.LRScheduler]

_default_optimizer = torch.optim.Adam
_default_lr_scheduler = partial(torch.optim.lr_scheduler.ConstantLR, factor=1)


class VIMALightningModule(pl.LightningModule):
    """Lighting module for training the VIMA model offline."""

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

        self._optimizer_partial_fn = optimizer_partial_fn
        self._lr_scheduler_partial_fn = lr_scheduler_partial_fn

        self._accuracy = PoseAccuracyMetric.from_config(
            max_num_pose_position_classes=max(
                policy.n_discrete_x_bins, policy.n_discrete_y_bins, policy.n_discrete_z_bins
            ),
            max_num_pose_rotation_classes=policy.n_discrete_rot_bins,
            ignore_index=self.ignore_target_index,
        )
        self._loss_per_axis = LossPerAxisPerActionMetric()
        self._examples_seen = SumMetric()

    @classmethod
    def from_wandb_run(
        cls,
        wandb_entity: str,
        wandb_project: str,
        wandb_run_id: str,
        checkpoint_save_dir: Path | str,
    ) -> Self:
        """Instantiate the model by getting the checkpoint from a wandb run."""
        checkpoint_save_dir = Path(checkpoint_save_dir)
        model_checkpoint_path = download_model_from_wandb(
            entity=wandb_entity,
            project=wandb_project,
            run_id=wandb_run_id,
            save_dir=checkpoint_save_dir,
        )
        return cls.load_from_checkpoint(model_checkpoint_path)

    def forward(self, batch: ModelInstance) -> dict[PoseActionType, MultiCategorical]:
        """Perform the forward on a batch of instances."""
        encoded_predicted_actions = self.policy.predict_action_token(
            encoded_prompt=batch.encoded_prompt,
            encoded_prompt_mask=batch.encoded_prompt_mask,
            embedded_observations=batch.embedded_observations,
            embedded_observations_mask=batch.embedded_observations_mask,
            embedded_actions=batch.embedded_actions,
        )

        predicted_actions_dists = self.policy.decode_action_token(encoded_predicted_actions)

        return predicted_actions_dists

    def training_step(
        self, batch: PreprocessedBatch, batch_idx: int  # noqa: ARG002
    ) -> torch.Tensor:
        """Perform a training step."""
        prepared_batch = self.embed_inputs(batch)
        predicted_actions = self.forward(prepared_batch)
        target_actions: dict[PoseActionType, torch.Tensor] = batch.actions.to_container()

        fine_grained_loss = compute_fine_grained_loss(
            predicted_actions, target_actions, ignore_target_index=self.ignore_target_index
        )
        loss = reduce_fine_grained_loss(fine_grained_loss)

        self._loss_per_axis.update(fine_grained_loss)
        self._update_accuracy(predicted_actions, target_actions)

        self.log(
            "train_loss", loss, prog_bar=True, logger=True, batch_size=len(batch), sync_dist=True
        )
        self._log_accuracy(
            split="train", prog_bar=True, logger=True, batch_size=len(batch), sync_dist=True
        )
        self._log_loss_per_axis(
            split="train", prog_bar=True, logger=True, batch_size=len(batch), sync_dist=True
        )

        # Log the total number of examples seen across all epochs (and doing it this way will
        # prevent the thing resetting every epoch)
        self._examples_seen.update(len(batch))
        self.log(
            "trainer/num_examples",
            self._examples_seen.compute(),
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        return loss

    def validation_step(
        self, batch: PreprocessedBatch, batch_idx: int  # noqa: ARG002
    ) -> torch.Tensor:
        """Perform a validation step (identical to training step)."""
        prepared_batch = self.embed_inputs(batch)
        predicted_actions = self.forward(prepared_batch)
        target_actions: dict[PoseActionType, torch.Tensor] = batch.actions.to_container()

        fine_grained_loss = compute_fine_grained_loss(
            predicted_actions, target_actions, ignore_target_index=self.ignore_target_index
        )
        loss = reduce_fine_grained_loss(fine_grained_loss)

        self._loss_per_axis.update(fine_grained_loss)
        self._update_accuracy(predicted_actions, target_actions)

        self.log(
            "val_loss", loss, prog_bar=True, logger=True, batch_size=len(batch), sync_dist=True
        )
        self._log_accuracy(
            split="val", prog_bar=True, logger=True, batch_size=len(batch), sync_dist=True
        )
        self._log_loss_per_axis(
            split="val", prog_bar=True, logger=True, batch_size=len(batch), sync_dist=True
        )

        return loss

    def configure_optimizers(self) -> Any:
        """Configure the optimizer and scheduler."""
        num_steps = self.trainer.estimated_stepping_batches
        optimizer = self._optimizer_partial_fn(self.parameters())
        scheduler = self._lr_scheduler_partial_fn(optimizer, total_steps=num_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def on_train_epoch_end(self) -> None:
        """Reset the accuracy metric at the end of the epoch."""
        self._accuracy.reset()
        self._loss_per_axis.reset()
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        """Reset the accuracy metric at the end of the epoch."""
        self._accuracy.reset()
        self._loss_per_axis.reset()
        return super().on_validation_epoch_end()

    def embed_inputs(self, batch: PreprocessedBatch) -> ModelInstance:
        """Embed a batch of instances and convert to the ModelInstance."""
        embedded_prompt, embedded_prompt_mask = self.policy.assemble_prompt(
            (batch.raw_prompts_token_type, batch.word_batch, batch.image_batch)
        )
        encoded_prompt = self.policy.encode_prompt(embedded_prompt, embedded_prompt_mask)
        embedded_observations, embedded_observations_mask = self.policy.embed_observation_token(
            batch.observations
        )
        embedded_actions = self.policy.embed_action_token(batch.actions)
        return ModelInstance(
            encoded_prompt=encoded_prompt,
            encoded_prompt_mask=embedded_prompt_mask,
            embedded_observations=embedded_observations,
            embedded_observations_mask=embedded_observations_mask,
            embedded_actions=embedded_actions,
        )

    @torch.no_grad()
    def _update_accuracy(
        self,
        predicted_actions: dict[PoseActionType, MultiCategorical],
        target_actions: dict[PoseActionType, torch.Tensor],
    ) -> None:
        """Update the accuracy metric for all the pose action types."""
        predicted_action_tokens: dict[str, torch.Tensor] = {
            pose_action_type: action_distribution.mode()
            for pose_action_type, action_distribution in predicted_actions.items()
        }
        self._accuracy.update(
            predicted_action_tokens, cast(dict[str, torch.Tensor], target_actions)
        )

    @torch.no_grad()
    def _log_accuracy(self, *, split: Literal["train", "val"], **log_dict_kwargs: Any) -> None:
        """Log the accuracy for the given split."""
        computed_acc = {
            f"{split}_{pose_name}_acc": acc for pose_name, acc in self._accuracy.compute().items()
        }
        self.log_dict(computed_acc, **log_dict_kwargs)

    @torch.no_grad()
    def _log_loss_per_axis(
        self, *, split: Literal["train", "val"], **log_dict_kwargs: Any
    ) -> None:
        """Log the loss per axis."""
        computed_loss = {
            f"{split}_{loss_name}_loss": loss
            for loss_name, loss in self._loss_per_axis.compute().items()
        }
        self.log_dict(computed_loss, **log_dict_kwargs)
