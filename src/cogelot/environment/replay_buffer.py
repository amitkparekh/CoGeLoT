import torch
from loguru import logger

from cogelot.data.collate import collate_variable_ndim_batch
from cogelot.structures.model import ModelInstance
from cogelot.structures.vima import Partition, Task


class ReplayBuffer:
    """Buffer for the current episode to allow replay."""

    def __init__(self) -> None:
        self._encoded_prompt: list[torch.Tensor] = []
        self._encoded_prompt_mask: list[torch.Tensor] = []
        self._embedded_observations: list[torch.Tensor] = []
        self._embedded_observation_masks: list[torch.Tensor] = []
        self._embedded_actions: list[torch.Tensor] = []
        self._task: Task | None = None
        self._partition: Partition | None = None

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

    def reset(self, task: Task, partition: Partition) -> None:
        """Reset the buffer."""
        logger.info("Resetting the state")

        self._encoded_prompt = []
        self._encoded_prompt_mask = []
        self._embedded_observations = []
        self._embedded_observation_masks = []
        self._embedded_actions = []

        self._task = task
        self._partition = partition
