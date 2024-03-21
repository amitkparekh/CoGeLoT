import inspect
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any, Self

import pytorch_lightning as pl
import torch
from loguru import logger

from cogelot.common.hf_models import (
    download_model_checkpoint,
    get_model_checkpoint_file_in_remote_repo_for_epoch,
)
from cogelot.common.hydra import instantiate_module_hparams_from_checkpoint
from cogelot.metrics.offline import OfflineMetrics
from cogelot.modules.policy import Policy
from cogelot.modules.tokenizers.pose_action import prepare_target_actions
from cogelot.nn.loss import compute_fine_grained_loss, reduce_fine_grained_loss
from cogelot.structures.model import ModelInstance, PreprocessedBatch
from cogelot.structures.vima import (
    N_DISCRETE_ROT_BINS,
    N_DISCRETE_X_BINS,
    N_DISCRETE_Y_BINS,
    N_DISCRETE_Z_BINS,
)

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
        should_shuffle_obj_per_observations: bool = False,
    ) -> None:
        super().__init__()
        self.policy = policy

        self._optimizer_partial_fn = optimizer_partial_fn
        self._lr_scheduler_partial_fn = lr_scheduler_partial_fn
        self._should_shuffle_obj_per_observations = should_shuffle_obj_per_observations

        self.training_metrics = OfflineMetrics(
            split_name_prefix="train",
            num_axes=14,
            max_num_classes=max(
                N_DISCRETE_X_BINS, N_DISCRETE_Y_BINS, N_DISCRETE_Z_BINS, N_DISCRETE_ROT_BINS
            ),
            ignore_index=self.ignore_target_index,
        )
        self.validation_metrics = OfflineMetrics(
            split_name_prefix="val",
            num_axes=14,
            max_num_classes=max(
                N_DISCRETE_X_BINS, N_DISCRETE_Y_BINS, N_DISCRETE_Z_BINS, N_DISCRETE_ROT_BINS
            ),
            ignore_index=self.ignore_target_index,
        )
        self.save_hyperparameters(ignore=["policy"])

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path, *args: Any, **kwargs: Any) -> Self:
        """Instantiate the model from the checkpoint."""
        logger.info(f"Loading model from checkpoint: `{checkpoint_path}`")
        try:
            return cls.load_from_checkpoint(checkpoint_path, *args, **kwargs)
        except TypeError:
            return cls.load_from_checkpoint(
                checkpoint_path,
                *args,
                **kwargs,
                **instantiate_module_hparams_from_checkpoint(checkpoint_path),
            )

    @classmethod
    def from_hf_repo(
        cls,
        wandb_run_id: str,
        hf_repo_id: str,
        epoch: int = -1,
    ) -> Self:
        """Instantiate the model by getting the checkpoint from a wandb run."""
        model_path_in_repo = get_model_checkpoint_file_in_remote_repo_for_epoch(
            repo_id=hf_repo_id, run_id=wandb_run_id, epoch=epoch
        )
        logger.info(f"Downloading model from remote path: `{model_path_in_repo}`")
        model_checkpoint_path = download_model_checkpoint(
            repo_id=hf_repo_id, file_path_in_repo=model_path_in_repo
        )
        return cls.from_checkpoint(model_checkpoint_path)

    def forward(self, batch: ModelInstance) -> torch.Tensor:
        """Perform the forward on a batch of instances."""
        return self.policy.predict_action_logits(
            # Shape: (bsz, prompt seq length, dim)
            encoded_prompt=batch.encoded_prompt,
            # Shape: (bsz, prompt seq length)
            encoded_prompt_mask=batch.encoded_prompt_mask,
            # Shape: (bsz, timesteps, max num objects, dim)
            encoded_observations=batch.encoded_observations,
            # Shape: (bsz, timesteps, max num objects)
            encoded_observations_mask=batch.encoded_observations_mask,
            # Shape: (bsz, timesteps, num axes, dim)
            encoded_actions=batch.encoded_actions,
            # Shape: (bsz, timesteps, num axes)
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

        discrete_target_actions = self.prepare_target_actions(batch)

        fine_grained_loss = compute_fine_grained_loss(
            predicted_actions,
            discrete_target_actions,
            ignore_target_index=self.ignore_target_index,
        )
        loss = reduce_fine_grained_loss(fine_grained_loss)

        self.training_metrics.update(
            fine_grained_loss=fine_grained_loss,
            predicted_actions=predicted_actions,
            target_actions=discrete_target_actions,
            tasks=batch.task,
        )

        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=len(batch))
        self.log_dict(
            self.training_metrics.compute(),
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
            sync_dist=True,
        )
        return loss

    def validation_step(
        self,
        batch: PreprocessedBatch,
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        """Perform a validation step (identical to training step)."""
        prepared_batch = self.embed_inputs(batch)
        predicted_actions = self.forward(prepared_batch)

        discrete_target_actions = self.prepare_target_actions(batch)

        fine_grained_loss = compute_fine_grained_loss(
            predicted_actions,
            discrete_target_actions,
            ignore_target_index=self.ignore_target_index,
        )
        loss = reduce_fine_grained_loss(fine_grained_loss)

        self.validation_metrics.update(
            fine_grained_loss=fine_grained_loss,
            predicted_actions=predicted_actions,
            target_actions=discrete_target_actions,
            tasks=batch.task,
        )

        self.log(
            "val_loss", loss, prog_bar=True, logger=True, batch_size=len(batch), sync_dist=True
        )
        self.log_dict(
            self.validation_metrics.compute(),
            prog_bar=True,
            logger=True,
            batch_size=len(batch),
            sync_dist=True,
        )
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
        """Reset the metrics at the end of the batch."""
        self.training_metrics.reset()
        return super().on_train_batch_end(*args, **kwargs)

    def on_validation_epoch_end(self) -> None:
        """Reset the metrics at the end of the epoch."""
        self.validation_metrics.reset()
        return super().on_validation_epoch_end()

    def embed_inputs(self, batch: PreprocessedBatch) -> ModelInstance:
        """Embed a batch of instances and convert to the ModelInstance."""
        embedded_prompt, embedded_prompt_mask = self.policy.embed_multimodal_prompt(
            (batch.raw_prompts_token_type, batch.word_batch, batch.image_batch)
        )
        encoded_prompt = self.policy.encode_prompt(embedded_prompt, embedded_prompt_mask)
        encoded_observations, embedded_observations_mask = self.policy.encode_observation_token(
            batch.observations,
            shuffle_obj_per_observation=self._should_shuffle_obj_per_observations,
        )
        encoded_actions, encoded_actions_mask = self.policy.encode_action_tokens(batch.actions)
        return ModelInstance(
            encoded_prompt=encoded_prompt,
            encoded_prompt_mask=embedded_prompt_mask,
            encoded_observations=encoded_observations,
            encoded_observations_mask=embedded_observations_mask,
            encoded_actions=encoded_actions,
            encoded_actions_mask=encoded_actions_mask,
        )

    @torch.no_grad()
    def prepare_target_actions(self, batch: PreprocessedBatch) -> torch.Tensor:
        """Prepare the target actions from a batch.

        This means making it a dictionary and discretizing them. Since these are target actions, we
        don't need to track the gradients for them either, so we can do this in a no_grad context.
        """
        return prepare_target_actions(
            batch.actions.to_container(),
            self.policy.pose_action_tokenizer,
            ignore_target_index=self.ignore_target_index,
        )
