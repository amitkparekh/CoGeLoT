from typing import cast

import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger
from numpy import typing as npt

from cogelot.environment import ReplayBuffer, VIMAEnvironment
from cogelot.metrics.online import OnlineEvaluationMetrics
from cogelot.models.training import VIMALightningModule
from cogelot.modules.instance_preprocessor import InstancePreprocessor
from cogelot.structures.common import Observation, PromptAssets
from cogelot.structures.model import EvaluationEpisode
from cogelot.structures.vima import (
    EndEffector,
    PoseActionType,
)
from vima.utils import DataDict, add_batch_dim


class EvaluationLightningModule(pl.LightningModule):
    """Lightning module for running multiple environments when evaluating the model."""

    def __init__(
        self,
        environment: VIMAEnvironment,
        model: VIMALightningModule,
        instance_preprocessor: InstancePreprocessor,
    ) -> None:
        super().__init__()
        self.environment = environment
        self.model = model
        self.preprocessor = instance_preprocessor
        self.pose_action_tokenizer = self.model.policy.pose_action_tokenizer
        self.buffer = ReplayBuffer()
        self.metric = OnlineEvaluationMetrics()

    def test_step(  # type: ignore[override]
        self,
        batch: EvaluationEpisode,
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        """Run a single episode online."""
        partition, task = batch
        self.buffer.reset(task, partition)
        self.environment.set_task(task, partition)

        # Resetting the environment returns the first observation
        observation = self.environment.reset()
        self.environment.render()

        # Create a VIMA instance from the environment, which parses all the metadata as we want it
        vima_instance = self.environment.create_vima_instance()
        # Add the prompt to the state
        self.add_prompt_to_buffer(
            prompt=vima_instance.prompt, prompt_assets=vima_instance.prompt_assets
        )

        # Run the task until it is done
        is_task_done = False
        is_task_successful = False

        while not is_task_done:
            logger.info(f"Taking step {len(self.buffer)}")

            # Add the observation to the state
            self.add_observation_to_buffer(
                observation=observation,
                object_ids=vima_instance.object_ids,
                end_effector=vima_instance.end_effector_type,
            )

            # Predict the next pose action token
            predicted_action_tokens = self.predict_next_pose_action_token()
            predicted_continuous_actions = (
                self.pose_action_tokenizer.convert_discrete_to_continuous(predicted_action_tokens)
            )

            self.add_pose_action_token_to_buffer(predicted_continuous_actions)

            # Convert the pose action token to the environment
            actions_for_env = self.pose_action_tokenizer.convert_discrete_token_to_environment(
                predicted_action_tokens
            )

            # Take a step in the environment
            (
                observation,
                is_task_done,
                is_task_successful,
            ) = self.take_step_in_environment(actions=actions_for_env)

        # Update the metric
        self.metric.update(
            partition,
            task,
            is_successful=is_task_successful,
            num_steps_taken=len(self.buffer),
        )
        self.log_dict(self.metric.compute(), logger=True, on_step=True, on_epoch=False)
        logger.info("Task finished")

    def take_step_in_environment(
        self, actions: dict[PoseActionType, npt.NDArray[np.float64]]
    ) -> tuple[Observation, bool, bool]:
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

        return observation, step_result.done or step_result.truncated, is_successful

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

        (
            embedded_prompt,
            embedded_prompt_mask,
        ) = self.model.policy.embed_multimodal_prompt(
            (raw_prompts_token_type, word_batch, image_batch)
        )
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
        ) = self.model.policy.encode_observation_token(prepared_observations)

        self.buffer.add_next_encoded_observation(encoded_observations, encoded_observation_masks)

    def add_pose_action_token_to_buffer(
        self, continuous_actions: dict[PoseActionType, torch.Tensor]
    ) -> None:
        """Add a pose action to the state."""
        # Shape: (num action tokens, embed dim) and (num action tokens)
        encoded_actions, encoded_actions_mask = self.model.policy.encode_action_tokens(
            continuous_actions
        )
        # We also need to add back in the timestep and batch dimension for consistency
        self.buffer.add_next_encoded_action(
            encoded_actions.unsqueeze(0).unsqueeze(0),
            encoded_actions_mask.unsqueeze(0).unsqueeze(0),
        )

    def predict_next_pose_action_token(self) -> dict[PoseActionType, torch.Tensor]:
        """Predict the next action tokens from the model."""
        logits = self.model(self.buffer.to_model_instance())
        normalised_logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        predicted_actions = normalised_logits.argmax(dim=-1)[:, 0, -1]
        predicted_action_tokens: dict[PoseActionType, torch.Tensor] = {
            "pose0_position": predicted_actions[:3],
            "pose0_rotation": predicted_actions[3:7],
            "pose1_position": predicted_actions[7:10],
            "pose1_rotation": predicted_actions[10:14],
        }
        return predicted_action_tokens
