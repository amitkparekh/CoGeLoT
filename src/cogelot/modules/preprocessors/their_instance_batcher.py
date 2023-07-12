import itertools
from typing import ClassVar, get_args

import torch
from torch.nn.utils.rnn import pad_sequence

from cogelot.modules.policy import Policy
from cogelot.structures.model import ModelInstance, PreprocessedInstance
from cogelot.structures.vima import PoseActionType
from vima.utils import DataDict


def collate_target_action_tokens(
    instances: list[PreprocessedInstance], ignore_index_value: int = -100
) -> dict[PoseActionType, torch.Tensor]:
    """Collate the target action tokens across the instances into a single representation.

    To ensure that we do not compute the loss over ignored values, we use a padding value of -100
    so that it is ignored by the cross-entropy loss.
    """
    actions: list[dict[PoseActionType, torch.Tensor]] = [
        instance.actions.to_container() for instance in instances
    ]

    collated_actions: dict[PoseActionType, torch.Tensor] = {}

    for pose_action_type in get_args(PoseActionType):
        collated_actions[pose_action_type] = pad_sequence(
            [action[pose_action_type] for action in actions],
            batch_first=False,
            padding_value=ignore_index_value,
        )

    return collated_actions


def stitch_observations_with_actions(
    observations: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
    """Stitch an instance of observations with actions."""
    observation_list = torch.split(observations, split_size_or_sections=1, dim=0)
    actions_list = torch.split(actions, split_size_or_sections=1, dim=0)

    zip_list = list(
        itertools.chain.from_iterable(zip(observation_list, actions_list, strict=False))
    )

    # If there are more observations than actions, then we need to add the remaining observations
    if len(zip_list) < len(observation_list):
        zip_list.append(observation_list[-1])

    stitched = torch.cat(zip_list, dim=0)

    return stitched


class TheirInstanceBatcher(torch.nn.Module):
    """Embed and prepare batches of instances."""

    views: ClassVar[set[str]] = {"front", "top"}
    default_mask_value: bool = False

    def __init__(self, policy: Policy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, instances: list[PreprocessedInstance]) -> ModelInstance:
        """Prepare a batch of instances."""
        encoded_prompts, prompt_masks = [], []
        encoded_observations, observation_masks = [], []
        encoded_actions = []

        for instance in instances:
            # Encode the prompt
            encoded_prompt, prompt_mask = self.encode_multimodal_prompt(
                instance.raw_prompts_token_type, instance.word_batch, instance.image_batch
            )
            encoded_prompts.append(encoded_prompt)
            prompt_masks.append(prompt_mask)

            # Encode the observations
            encoded_observation, observation_mask = self.encode_observations(instance.observations)
            encoded_observations.append(encoded_observation)
            observation_masks.append(observation_mask)

            # Encode the actions history
            encoded_action = self.encode_actions(instance.actions)
            encoded_actions.append(encoded_action)

        prompt_batch = pad_sequence(encoded_prompts, batch_first=True)
        prompt_mask_batch = pad_sequence(
            prompt_masks, batch_first=True, padding_value=self.default_mask_value
        )
        observation_batch, observation_mask_batch = self.batch_observations(
            encoded_observations, observation_masks
        )
        actions_batch = pad_sequence(encoded_actions, batch_first=True)

        return ModelInstance(
            embedded_prompt=prompt_batch,
            embedded_prompt_mask=prompt_mask_batch,
            embedded_observations=observation_batch,
            embedded_observations_mask=observation_mask_batch,
            embedded_actions=actions_batch,
        )

    def encode_multimodal_prompt(
        self,
        raw_prompt_token_type: list[list[int]],
        word_batch: torch.Tensor,
        image_batch: DataDict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the multimodal prompt."""
        encoded_prompt, prompt_mask = self.policy.assemble_prompt(
            (raw_prompt_token_type, word_batch, image_batch)
        )
        return encoded_prompt.squeeze(0), prompt_mask.squeeze(0)

    def encode_observations(self, observations: DataDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the observations."""
        encoded_obs, obs_mask = self.policy.embed_observation_token(observations)
        encoded_obs = encoded_obs.transpose(0, 1).squeeze(0)
        obs_mask = obs_mask.transpose(0, 1).squeeze(0)

        return encoded_obs, obs_mask

    def encode_actions(self, actions: DataDict) -> torch.Tensor:
        """Encode the actions into a tensor."""
        encoded_actions = self.policy.embed_action_token(actions)
        # If there are 3 dims, just get it down to 2
        # if encoded_actions.ndim == 3:
        #     encoded_actions = encoded_actions.squeeze(0)
        return encoded_actions

    def batch_observations(
        self, observations: list[torch.Tensor], masks: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create a batch of observations."""
        max_num_obs = max([obs.shape[0] for obs in observations])
        max_num_obj = max([obs.shape[1] for obs in observations])

        updated_obs = []
        updated_masks = []

        for obs, mask in zip(observations, masks, strict=True):
            num_obs_delta = max_num_obs - obs.shape[0]
            num_obj_delta = max_num_obj - obs.shape[1]

            padding = (0, num_obj_delta, 0, num_obs_delta)
            new_obs = torch.nn.functional.pad(obs, (0, 0, *padding))
            new_mask = torch.nn.functional.pad(mask, padding, value=self.default_mask_value)

            updated_obs.append(new_obs)
            updated_masks.append(new_mask)

        return torch.stack(updated_obs, dim=0), torch.stack(updated_masks, dim=0)
