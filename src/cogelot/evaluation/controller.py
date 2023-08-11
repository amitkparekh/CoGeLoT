from typing import cast

import numpy as np
import torch
from loguru import logger
from numpy import typing as npt

from cogelot.data.collate import collate_variable_ndim_batch
from cogelot.data.preprocess import InstancePreprocessor
from cogelot.evaluation.env_wrappers import VIMAEnvironment
from cogelot.models.vima import VIMALightningModule
from cogelot.structures.common import Assets, Observation
from cogelot.structures.model import ModelInstance
from cogelot.structures.vima import ActionBounds, PoseActionType
from vima.utils import DataDict, add_batch_dim, any_slice


class OnlineInstanceState:
    """State buffer for the OnlineEvaluationController."""

    def __init__(self) -> None:
        self._encoded_prompt: list[torch.Tensor] = []
        self._encoded_prompt_mask: list[torch.Tensor] = []
        self._embedded_observations: list[torch.Tensor] = []
        self._embedded_observation_masks: list[torch.Tensor] = []
        self._embedded_actions: list[torch.Tensor] = []

    def __len__(self) -> int:
        """Get the number of steps taken."""
        return self.num_actions

    @property
    def num_observations(self) -> int:
        """Get the number of observations."""
        return len(self._embedded_observations)

    @property
    def num_actions(self) -> int:
        """Get the number of actions."""
        return len(self._embedded_actions)

    @property
    def encoded_prompt(self) -> torch.Tensor:
        """Get the embedded prompt."""
        return self._encoded_prompt[0]

    @encoded_prompt.setter
    def encoded_prompt(self, encoded_prompt: torch.Tensor) -> None:
        """Set the embedded prompt."""
        if self._encoded_prompt:
            raise ValueError("Cannot set the embedded prompt twice. Need to reset buffer first.")
        self._encoded_prompt.append(encoded_prompt)

    @property
    def encoded_prompt_mask(self) -> torch.Tensor:
        """Get the embedded prompt mask."""
        return self._encoded_prompt_mask[0]

    @encoded_prompt_mask.setter
    def encoded_prompt_mask(self, encoded_prompt_mask: torch.Tensor) -> None:
        """Set the embedded prompt."""
        if self._encoded_prompt_mask:
            raise ValueError("Cannot set the embedded prompt twice. Need to reset buffer first.")
        self._encoded_prompt_mask.append(encoded_prompt_mask)

    @property
    def embedded_observations(self) -> torch.Tensor:
        """Get the embedded observations."""
        # Firstly, we need to remove the batch dim since this has a batch size of 1, and we need to
        # remove the observation dim, since each tensor has only one observation. Each tensor is
        # now 2D
        observations = [tensor.squeeze(0).squeeze(0) for tensor in self._embedded_observations]

        # Then we can use the collate function to make sure it is all padded correctly, and that
        # gives us the observation dimension back, making it 3D
        collated_observations = collate_variable_ndim_batch(observations)

        # Then we just need to add the batch dimension back to make it 4D
        collated_observations = collated_observations.unsqueeze(0)
        return collated_observations

    @property
    def embedded_observation_masks(self) -> torch.Tensor:
        """Get the embedded observation masks."""
        # Firstly, remove the batch and observation dimensions since they are just 1, leaving us a
        # 1D tensor
        masks = [tensor.squeeze(0).squeeze(0) for tensor in self._embedded_observation_masks]

        # Then we can use the collate function to make sure it is all padded correctly
        collated_masks = collate_variable_ndim_batch(masks)

        # Then we just need to add the batch dim back
        collated_masks = collated_masks.unsqueeze(0)
        return collated_masks

    @property
    def embedded_actions(self) -> torch.Tensor | None:
        """Get the embedded actions."""
        if not self._embedded_actions:
            return None

        # Remove the batch dims and the actions dim
        actions = [tensor.squeeze(0).squeeze(0) for tensor in self._embedded_actions]

        # Then we can use the collate function to make sure it is all padded correctly
        collated_actions = collate_variable_ndim_batch(actions)

        # Then we just need to add the batch dim back
        collated_actions = collated_actions.unsqueeze(0)
        return collated_actions

    def add_next_embedded_observation(
        self, embedded_observation: torch.Tensor, embddded_observation_mask: torch.Tensor
    ) -> None:
        """Add an embedded observation to the buffer."""
        self._embedded_observations.append(embedded_observation)
        self._embedded_observation_masks.append(embddded_observation_mask)

    def add_next_embedded_action(self, embedded_action: torch.Tensor) -> None:
        """Add an embedded action to the buffer."""
        self._embedded_actions.append(embedded_action)

    def to_model_instance(self) -> ModelInstance:
        """Convert the state into a ModelInstance."""
        return ModelInstance(
            encoded_prompt=self.encoded_prompt,
            encoded_prompt_mask=self.encoded_prompt_mask,
            embedded_observations=self.embedded_observations,
            embedded_observations_mask=self.embedded_observation_masks,
            embedded_actions=self.embedded_actions,
        )

    def reset(self) -> None:
        """Reset the buffer."""
        logger.info("Resetting the state")

        self._encoded_prompt = []
        self._encoded_prompt_mask = []
        self._embedded_observations = []
        self._embedded_observation_masks = []
        self._embedded_actions = []


@torch.inference_mode()
class OnlineEvaluationController:
    """Run the online evaluation for the given environment."""

    def __init__(
        self,
        environment: VIMAEnvironment,
        model: VIMALightningModule,
        instance_preprocessor: InstancePreprocessor,
    ) -> None:
        self.environment = environment
        self.instance_preprocessor = instance_preprocessor
        self.model = model
        self.state = OnlineInstanceState()

    def run_task(self) -> None:
        """Run the task."""
        self.state.reset()
        # Resetting the environment returns the first observation
        observation = self.environment.reset()
        self.environment.render()

        # Create a VIMA instance from the environment, which parses all the metadata as we want it
        vima_instance = self.environment.create_vima_instance()

        # Add the prompt to the state
        self.add_prompt_to_state(
            prompt=vima_instance.prompt, prompt_assets=vima_instance.prompt_assets
        )

        # Run the task until it is done
        is_task_done = False
        is_task_successful = False

        while not is_task_done:
            # Add the observation to the state
            self.add_observation_to_state(
                observation=observation,
                object_ids=vima_instance.object_ids,
                end_effector=vima_instance.end_effector_type,
            )

            # Predict the next pose action token
            predicted_action_tokens = self.predict_next_pose_action_token()

            self.add_pose_action_token_to_state(pose_action_tokens=predicted_action_tokens)

            # Convert the pose action token to the environment
            actions_for_env = self.convert_pose_action_token_to_environment(
                action_token=predicted_action_tokens, action_bounds=vima_instance.action_bounds
            )

            # Take a step in the environment
            observation, is_task_done, is_task_successful = self.take_step_in_environment(
                actions=actions_for_env
            )

        logger.info(f"Task is done: {is_task_done}")
        logger.info(f"Task is successful: {is_task_successful}")

    def add_prompt_to_state(self, prompt: str, prompt_assets: Assets) -> None:
        """Prepare and encode the prompt."""
        raw_prompts_token_type, word_batch, image_batch = (
            self.instance_preprocessor.prepare_prompt(
                prompt=prompt,
                prompt_assets=prompt_assets.dict()["__root__"],
                object_ids_from_prompt_assets=prompt_assets.all_object_ids,
            )
        )
        # Need to add the batch dimension to the word batch
        word_batch = word_batch.unsqueeze(0)
        embedded_prompt, embedded_prompt_mask = self.model.policy.assemble_prompt(
            (raw_prompts_token_type, word_batch, image_batch)
        )
        encoded_prompt = self.model.policy.encode_prompt(embedded_prompt, embedded_prompt_mask)

        self.state.encoded_prompt = encoded_prompt
        self.state.encoded_prompt_mask = embedded_prompt_mask

    def add_observation_to_state(
        self, observation: Observation, object_ids: set[int], end_effector: str
    ) -> None:
        """Prepare and embed the observations."""
        prepared_observations = self.instance_preprocessor.prepare_observations(
            observations=[observation],
            object_ids=object_ids,
            end_effector=end_effector,
        )

        # For some reason, the batch dimension is not added to the observations, so we need to add
        # it in
        prepared_observations = cast(DataDict, prepared_observations.map_structure(add_batch_dim))
        embedded_observations, embedded_observation_masks = (
            self.model.policy.embed_observation_token(prepared_observations)
        )

        self.state.add_next_embedded_observation(embedded_observations, embedded_observation_masks)

    def add_pose_action_token_to_state(
        self, pose_action_tokens: dict[PoseActionType, torch.Tensor]
    ) -> None:
        """Add a pose action to the state."""
        encoded_pose_actions = self.model.policy.embed_action_token(pose_action_tokens)
        self.state.add_next_embedded_action(encoded_pose_actions)

    def predict_next_pose_action_token(self) -> dict[PoseActionType, torch.Tensor]:
        """Predict the next action tokens from the model."""
        model_instance = self.state.to_model_instance()
        predicted_actions = self.model.forward(model_instance)
        predicted_action_tokens: dict[PoseActionType, torch.Tensor] = {
            pose_action_type: action_distribution.mode()
            for pose_action_type, action_distribution in predicted_actions.items()
        }
        return predicted_action_tokens

    def take_step_in_environment(
        self, actions: dict[PoseActionType, npt.NDArray[np.float64]]
    ) -> tuple[Observation, bool, bool]:
        """Take a step in the environment, and return the next observation."""
        obs, _, is_done, task_info = self.environment.step(actions)

        assert isinstance(obs, dict)
        observation = Observation.parse_obj({"index": self.state.num_observations, **obs})

        is_successful = task_info["success"]
        assert isinstance(is_successful, bool)

        return observation, is_done, is_successful

    def convert_pose_action_token_to_environment(
        self, action_token: dict[PoseActionType, torch.Tensor], action_bounds: ActionBounds
    ) -> dict[PoseActionType, npt.NDArray[np.float64]]:
        """Convert the pose action tokens to the environment."""
        actions = self.model.policy.de_discretize_actions(action_token)

        tensor_device = actions["pose0_position"].device
        action_bounds_low = torch.from_numpy(action_bounds.low).to(tensor_device)
        action_bounds_high = torch.from_numpy(action_bounds.high).to(tensor_device)

        actions["pose0_position"] = (
            actions["pose0_position"] * (action_bounds_high - action_bounds_low)
            + action_bounds_low
        )
        actions["pose1_position"] = (
            actions["pose1_position"] * (action_bounds_high - action_bounds_low)
            + action_bounds_low
        )
        actions["pose0_position"] = torch.clamp(
            actions["pose0_position"], min=action_bounds_low, max=action_bounds_high
        )
        actions["pose1_position"] = torch.clamp(
            actions["pose1_position"], min=action_bounds_low, max=action_bounds_high
        )
        actions["pose0_rotation"] = actions["pose0_rotation"] * 2 - 1
        actions["pose1_rotation"] = actions["pose1_rotation"] * 2 - 1
        actions["pose0_rotation"] = torch.clamp(actions["pose0_rotation"], min=-1, max=1)
        actions["pose1_rotation"] = torch.clamp(actions["pose1_rotation"], min=-1, max=1)

        actions_numpy = {k: v.cpu().numpy() for k, v in actions.items()}
        actions_numpy = any_slice(actions_numpy, np.s_[0, 0])

        return cast(dict[PoseActionType, npt.NDArray[np.float64]], actions_numpy)
