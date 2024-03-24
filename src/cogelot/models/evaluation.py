import sys
from typing import cast

import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger
from numpy import typing as npt

from cogelot.data.transforms import NoopTransform, VIMAInstanceTransform
from cogelot.environment import ReplayBuffer, VIMAEnvironment
from cogelot.environment.vima import GetObservationError
from cogelot.metrics.online import EvaluationEpisodeTracker, OnlineEvaluationMetrics
from cogelot.models.training import VIMALightningModule
from cogelot.modules.instance_preprocessor import InstancePreprocessor
from cogelot.modules.tokenizers.pose_action import is_action_pointless
from cogelot.structures.common import Observation, PromptAssets
from cogelot.structures.model import EvaluationEpisode
from cogelot.structures.vima import (
    Difficulty,
    EndEffector,
    Partition,
    PoseActionType,
    Task,
    VIMAInstance,
)
from vima.utils import DataDict, add_batch_dim
from vima_bench.env.base import MovementFailedError

NUM_AXES = 14
MAX_TIMESTEPS = 20


class EvaluationLightningModule(pl.LightningModule):
    """Lightning module for running multiple environments when evaluating the model."""

    def __init__(
        self,
        environment: VIMAEnvironment,
        model: VIMALightningModule,
        instance_preprocessor: InstancePreprocessor,
        vima_instance_transform: VIMAInstanceTransform | None = None,
        *,
        difficulty: Difficulty = "easy",
        should_stop_on_first_success: bool = True,
        max_timesteps: int = MAX_TIMESTEPS,
        disable_prompt_text: bool = False,
        disable_prompt_visual: bool = False,
        should_shuffle_obj_per_observations: bool = False,
    ) -> None:
        if vima_instance_transform is None:
            vima_instance_transform = NoopTransform()

        super().__init__()
        self.environment = environment
        self.model = model
        self.preprocessor = instance_preprocessor
        self.pose_action_tokenizer = self.model.policy.pose_action_tokenizer

        self.buffer = ReplayBuffer()
        self._metric = OnlineEvaluationMetrics()
        self._episode_tracker = EvaluationEpisodeTracker()

        self._vima_instance_transform = vima_instance_transform
        self._should_stop_on_first_success = should_stop_on_first_success
        self._max_timesteps = max_timesteps

        self._disable_prompt_text = disable_prompt_text
        self._disable_prompt_visual = disable_prompt_visual
        self._should_shuffle_obj_per_observations = should_shuffle_obj_per_observations
        self._difficulty: Difficulty = difficulty

    def test_step(
        self,
        batch: EvaluationEpisode,
        batch_idx: int,
        *,
        is_retry: bool = False,
    ) -> None:
        """Run a single episode online."""
        partition, task = batch
        self.reset_environment(task=task, partition=partition)

        # Create a VIMA instance from the environment, which parses all the metadata as we want it
        vima_instance = self.environment.create_vima_instance(partition)
        # Transform the instance as desired
        vima_instance = self._vima_instance_transform(vima_instance)
        try:
            self.run_vima_instance(vima_instance)
        except GetObservationError as err:
            logger.error(
                "Something went wrong when getting the observation. Resetting and trying again."
            )
            if not is_retry:
                return self.test_step(batch, batch_idx, is_retry=True)

            raise GetObservationError("Already retried once, crashing out") from err

    def on_test_start(self) -> None:
        """When the test starts, prefix all the logs with the current rank."""
        logger.remove()
        logger_message_prefix = f"[rank {self.local_rank}]"
        logger.add(
            sys.stderr,
            format=logger_message_prefix
            + " <green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        )

    def on_test_end(self) -> None:
        """Log the episode details."""
        logger.remove()
        logger.add(sys.stderr)
        self._episode_tracker.upload_table()

    def reset_environment(self, task: Task, partition: Partition) -> None:
        """Reset the environment."""
        logger.debug("Resetting environment")

        self.buffer.reset()
        self.environment.set_task(task, partition, self._difficulty)

        self.environment.reset()
        self.environment.render()

    def run_vima_instance(self, vima_instance: VIMAInstance) -> None:
        """Run the current instance in the environment."""
        observation = self.environment.get_first_observation()

        # Add the observation to the state
        self.add_observation_to_buffer(
            observation=observation,
            object_ids=vima_instance.object_ids,
            end_effector=vima_instance.end_effector_type,
        )

        # Add the prompt to the state
        self.add_prompt_to_buffer(
            prompt=vima_instance.prompt, prompt_assets=vima_instance.prompt_assets
        )

        # Run the task until the model thinks it is done
        while len(self.buffer) < self._max_timesteps:
            logger.info(f"Taking step {len(self.buffer)}")

            # Predict the next pose action token
            predicted_discrete_action_tokens = self.predict_next_pose_action_token()
            predicted_continuous_actions = (
                self.pose_action_tokenizer.convert_discrete_to_continuous(
                    predicted_discrete_action_tokens
                )
            )

            self.add_continuous_actions_to_buffer(predicted_continuous_actions)

            # Convert the pose action token to the environment
            actions_for_env = self.pose_action_tokenizer.convert_continuous_token_to_environment(
                predicted_continuous_actions
            )

            if is_action_pointless(actions_for_env):
                logger.info("Model returned pointless action; terminating early")
                break

            # Take a step in the environment
            try:
                observation, is_task_successful = self.take_action_in_environment(
                    actions=actions_for_env
                )
            except MovementFailedError:
                logger.error("Movement failed; terminating early")
                self.buffer.update_success_tracker(is_successful=False)
                break

            # Add the observation to the state
            self.add_observation_to_buffer(
                observation=observation,
                object_ids=vima_instance.object_ids,
                end_effector=vima_instance.end_effector_type,
            )

            self.buffer.update_success_tracker(is_successful=is_task_successful)
            if is_task_successful and self._should_stop_on_first_success:
                logger.info("Task successful; terminating early")
                break

        # Update the metric
        self._metric.update(
            vima_instance.partition,
            vima_instance.task,
            success_tracker_per_step=self.buffer.success_per_step,
            num_steps_taken=len(self.buffer),
        )

        logger.debug("Updating all the episode details.")
        vima_instance = self.buffer.update_vima_instance(vima_instance)
        self._episode_tracker.update(vima_instance=vima_instance)

        logger.debug("Logging all the episode details.")
        self.log_dict(self._metric.compute(), logger=True, on_step=True, on_epoch=False)
        logger.info("Task finished")

    def take_action_in_environment(
        self, actions: dict[PoseActionType, npt.NDArray[np.float32]]
    ) -> tuple[Observation, bool]:
        """Take a step in the environment, and return the next observation."""
        logger.debug("Taking step in the environment")
        step_result = self.environment.step(actions)

        logger.debug("Parsing response from environment")

        assert isinstance(step_result.observation, dict)
        observation = Observation.model_validate(
            {"index": self.buffer.num_observations, **step_result.observation}
        )

        is_successful = step_result.task_info["success"]
        assert isinstance(is_successful, bool)

        return observation, is_successful

    def add_prompt_to_buffer(self, prompt: str, prompt_assets: PromptAssets) -> None:
        """Prepare and encode the prompt."""
        (
            raw_prompts_token_type,
            word_batch,
            image_batch,
        ) = self.preprocessor.prepare_prompt(
            prompt=prompt,
            prompt_assets=prompt_assets,
            object_ids_from_prompt_assets=None,
        )

        # Update devices
        word_batch = word_batch.to(self.device)
        image_batch = image_batch.to_torch_tensor(device=self.device)

        # The following functions assume that there is a batch dimension for the word and image
        # batch, therefore we are going to need to add one.
        word_batch = cast(torch.Tensor, add_batch_dim(word_batch))
        image_batch = cast(DataDict, add_batch_dim(image_batch))

        # Create the text mask (inverting happens during assembly, so True means DO NOT MASK at
        # this point in the flow)
        text_mask = torch.ones_like(word_batch, dtype=torch.bool)

        # Do some optional disabling of modalities if need be
        word_batch, text_mask = self._maybe_disable_words(word_batch, text_mask)
        image_batch = self._maybe_disable_visuals(image_batch)

        # The image_batch is none when we are using the textual instance transformation. Otherwise
        # it shouldn't be none.
        if image_batch:
            (
                embedded_prompt,
                embedded_prompt_mask,
            ) = self.model.policy.embed_multimodal_prompt(
                (raw_prompts_token_type, word_batch, image_batch), text_mask=text_mask
            )
        else:
            embedded_prompt = self.model.policy.prompt_embedding(word_batch)
            # At this point, the mask is inverted to follow "torch" meanings
            embedded_prompt_mask = ~text_mask

        encoded_prompt = self.model.policy.encode_prompt(embedded_prompt, embedded_prompt_mask)
        self.buffer.encoded_prompt = encoded_prompt
        self.buffer.encoded_prompt_mask = embedded_prompt_mask

    def add_observation_to_buffer(
        self, observation: Observation, object_ids: set[int], end_effector: EndEffector
    ) -> None:
        """Prepare and embed the observations."""
        prepared_observations = self.preprocessor.prepare_observations(
            observations=[observation],
            object_ids=object_ids,
            end_effector=end_effector,
        )

        # Update the device
        prepared_observations = cast(
            DataDict, prepared_observations.to_torch_tensor(device=self.device)
        )

        # For some reason, the batch dimension is not added to the observations, so we need to add
        # it in
        prepared_observations = cast(DataDict, prepared_observations.map_structure(add_batch_dim))

        (
            encoded_observations,
            encoded_observation_masks,
        ) = self.model.policy.encode_observation_token(
            prepared_observations,
            shuffle_obj_per_observation=self._should_shuffle_obj_per_observations,
        )

        self.buffer.add_next_encoded_observation(encoded_observations, encoded_observation_masks)
        self.buffer.add_observation(observation)

    def add_continuous_actions_to_buffer(
        self, continuous_actions: dict[PoseActionType, torch.Tensor]
    ) -> None:
        """Add the continuous actions to the buffer."""
        # Add the actions to the buffer before we do anything else to them
        self.buffer.add_action(continuous_actions)

        # We also need to add back in the timestep and batch dimension for consistency and thats
        # what the model wants.
        continuous_actions = {
            pose_action_type: action.unsqueeze(0).unsqueeze(0)
            for pose_action_type, action in continuous_actions.items()
        }

        encoded_actions, encoded_actions_mask = self.model.policy.encode_action_tokens(
            continuous_actions
        )
        self.buffer.add_next_encoded_action(encoded_actions, encoded_actions_mask)

    def predict_next_pose_action_token(self) -> dict[PoseActionType, torch.Tensor]:
        """Predict the next discrete action tokens from the model."""
        logits = self.model(self.buffer.to_model_instance())
        predicted_actions = logits.softmax(dim=-1).argmax(dim=-1)[:, 0, -1]
        split_sizes = [3, 4, 3, 4] if predicted_actions.shape[-1] == NUM_AXES else [2, 4, 2, 4]
        split_predicted_actions = predicted_actions.split(split_sizes, dim=-1)
        predicted_action_tokens: dict[PoseActionType, torch.Tensor] = {
            "pose0_position": split_predicted_actions[0],
            "pose0_rotation": split_predicted_actions[1],
            "pose1_position": split_predicted_actions[2],
            "pose1_rotation": split_predicted_actions[3],
        }
        return predicted_action_tokens

    def _maybe_disable_words(
        self, word_batch: torch.Tensor, text_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """If desired, disable the text modality."""
        if not self._disable_prompt_text:
            return word_batch, text_mask

        return word_batch, torch.zeros_like(text_mask, dtype=torch.bool)

    def _maybe_disable_visuals(self, image_batch: DataDict | None) -> DataDict | None:
        """If desired, disable the visual modality."""
        if not self._disable_prompt_visual or image_batch is None:
            return image_batch

        assert "mask" in image_batch
        masks = cast(dict[str, torch.Tensor], image_batch["mask"])
        image_batch["mask"] = {
            view: torch.zeros_like(tensor, dtype=torch.bool) for view, tensor in masks.items()
        }
        return image_batch
