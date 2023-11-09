import inspect
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any, Literal, Self

import pytorch_lightning as pl
import torch

from cogelot.common.hydra import instantiate_module_hparams_from_checkpoint
from cogelot.common.wandb import download_model_from_wandb
from cogelot.modules.metrics import (
    TrainingMetrics,
)
from cogelot.modules.policy import Policy
from cogelot.nn.loss import compute_fine_grained_loss, reduce_fine_grained_loss
from cogelot.structures.model import ModelInstance, PreprocessedBatch
from cogelot.structures.vima import (
    N_DISCRETE_ROT_BINS,
    N_DISCRETE_X_BINS,
    N_DISCRETE_Y_BINS,
    N_DISCRETE_Z_BINS,
    PoseActionType,
)
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

        self.metrics = TrainingMetrics(
            max_num_classes=max(
                N_DISCRETE_X_BINS, N_DISCRETE_Y_BINS, N_DISCRETE_Z_BINS, N_DISCRETE_ROT_BINS
            ),
            ignore_index=self.ignore_target_index,
        )
        self.save_hyperparameters(logger=False)

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
        try:
            return cls.load_from_checkpoint(model_checkpoint_path)
        except TypeError:
            return cls.load_from_checkpoint(
                model_checkpoint_path,
                **instantiate_module_hparams_from_checkpoint(model_checkpoint_path),
            )

    def forward(self, batch: ModelInstance) -> dict[PoseActionType, MultiCategorical]:
        """Perform the forward on a batch of instances."""
        return self.policy.predict_actions(
            encoded_prompt=batch.encoded_prompt,
            encoded_prompt_mask=batch.encoded_prompt_mask,
            encoded_observations=batch.encoded_observations,
            encoded_observations_mask=batch.encoded_observations_mask,
            encoded_actions=batch.encoded_actions,
            encoded_actions_mask=batch.encoded_actions_mask,
        )

    def training_step(
        self,
        batch: PreprocessedBatch,
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        """Perform a training step."""
        prepared_batch = self.embed_inputs(batch)
        predicted_actions = self.forward(prepared_batch)
        target_actions: dict[PoseActionType, torch.Tensor] = batch.actions.to_container()
        discrete_target_actions = self.policy.tokenize_continuous_actions(target_actions)

        fine_grained_loss = compute_fine_grained_loss(
            predicted_actions,
            discrete_target_actions,
            ignore_target_index=self.ignore_target_index,
        )
        loss = reduce_fine_grained_loss(fine_grained_loss)

        self.metrics.update_loss(fine_grained_loss, tasks=batch.task)
        self.metrics.update_accuracy(predicted_actions, discrete_target_actions, tasks=batch.task)
        # Log the total number of examples seen across all epochs (and doing it this way will
        # prevent the thing resetting every epoch)
        self.metrics.update_examples_seen(len(batch))

        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=len(batch))
        self._log_metrics(split="train", prog_bar=True, logger=True, batch_size=len(batch))

        return loss

    def validation_step(
        self,
        batch: PreprocessedBatch,
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        """Perform a validation step (identical to training step)."""
        prepared_batch = self.embed_inputs(batch)
        predicted_actions = self.forward(prepared_batch)
        target_actions: dict[PoseActionType, torch.Tensor] = batch.actions.to_container()
        discrete_target_actions = self.policy.tokenize_continuous_actions(target_actions)

        fine_grained_loss = compute_fine_grained_loss(
            predicted_actions,
            discrete_target_actions,
            ignore_target_index=self.ignore_target_index,
        )
        loss = reduce_fine_grained_loss(fine_grained_loss)

        self.metrics.update_loss(fine_grained_loss, tasks=batch.task)
        self.metrics.update_accuracy(predicted_actions, discrete_target_actions, tasks=batch.task)

        # There may be a PossibleUserWarning about needing to use `sync_dist=True` here. However,
        # as we are using torchmetrics, we should not/do not need to be adding that flag. For more,
        # see: https://github.com/Lightning-AI/lightning/discussions/6501#discussioncomment-553152
        # and https://lightning.ai/docs/pytorch/stable/accelerators/accelerator_prepare.html#synchronize-validation-and-test-logging
        self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=len(batch))
        self._log_metrics(split="val", prog_bar=True, logger=True, batch_size=len(batch))

        return loss

    def configure_optimizers(self) -> Any:
        """Configure the optimizer and scheduler."""
        optimizer = self._optimizer_partial_fn(self.parameters())
        scheduler_kwargs = (
            {"total_steps": self.trainer.estimated_stepping_batches}
            if "total_steps" in inspect.signature(self._lr_scheduler_partial_fn).parameters
            else {}
        )
        scheduler = self._lr_scheduler_partial_fn(optimizer, **scheduler_kwargs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def on_train_batch_end(self, *args: Any, **kwargs: Any) -> None:
        """Reset the accuracy metric at the end of the epoch."""
        self.metrics.reset()
        return super().on_train_batch_end(*args, **kwargs)

    def on_validation_epoch_end(self) -> None:
        """Reset the accuracy metric at the end of the epoch."""
        self.metrics.reset()
        return super().on_validation_epoch_end()

    def embed_inputs(self, batch: PreprocessedBatch) -> ModelInstance:
        """Embed a batch of instances and convert to the ModelInstance."""
        encoded_prompt, encoded_prompt_mask = self.policy.embed_multimodal_prompt(
            (batch.raw_prompts_token_type, batch.word_batch, batch.image_batch)
        )
        encoded_observations, embedded_observations_mask = self.policy.encode_observation_token(
            batch.observations
        )
        encoded_actions, encoded_actions_mask = self.policy.encode_action_tokens(batch.actions)
        return ModelInstance(
            encoded_prompt=encoded_prompt,
            encoded_prompt_mask=encoded_prompt_mask,
            encoded_observations=encoded_observations,
            encoded_observations_mask=embedded_observations_mask,
            encoded_actions=encoded_actions,
            encoded_actions_mask=encoded_actions_mask,
        )

    @torch.no_grad()
    def _log_metrics(self, *, split: Literal["train", "val"], **log_dict_kwargs: Any) -> None:
        """Log the accuracy for the given split."""
        self.log_dict(self.metrics.compute(split=split), **log_dict_kwargs)
