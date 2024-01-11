import abc
from collections.abc import Mapping
from typing import Any, Self, cast

import torch
from einops import rearrange

from cogelot.modules.tokenizers.pose_action import PoseActionTokenizer
from cogelot.structures.vima import PoseActionType
from vima import nn as vnn


def _remove_zth_dimension(
    module: torch.nn.Module,  # noqa: ARG001
    *args: tuple[Any, ...],
) -> dict[PoseActionType, torch.Tensor]:
    """Remove the zth dimension from the continuous actions.

    This is because the zth dimension is always 0 for the actions we care about.
    """
    continuous_actions: dict[PoseActionType, torch.Tensor] = args[0][0]
    return {
        "pose0_position": continuous_actions["pose0_position"][..., :2],
        "pose0_rotation": continuous_actions["pose0_rotation"],
        "pose1_position": continuous_actions["pose1_position"][..., :2],
        "pose1_rotation": continuous_actions["pose1_rotation"],
    }


class ActionEncoder(abc.ABC, torch.nn.Module):
    """Embed actions from their continuous form to something that is useful for the model."""

    @property
    @abc.abstractmethod
    def num_action_tokens_per_timestep(self) -> int:
        """The number of action tokens per timestep."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, continuous_actions: dict[PoseActionType, torch.Tensor]) -> torch.Tensor:
        """Embed the continuous actions."""
        raise NotImplementedError


class VIMAContinuousActionEmbedder(ActionEncoder):
    """Encode actions from their continuous form.

    First normalize the continuous actions and then encode them through an MLP.

    This follows the VIMA way of doing things.
    """

    def __init__(
        self,
        *,
        pose_action_tokenizer: PoseActionTokenizer,
        embedder_per_pose_action: Mapping[PoseActionType, vnn.ContinuousActionEmbedding],
        post_layer: torch.nn.Linear | torch.nn.Identity,
    ) -> None:
        super().__init__()
        self._pose_action_tokenizer = pose_action_tokenizer
        self._embedder_per_pose_action = torch.nn.ModuleDict(
            cast(Mapping[str, torch.nn.Module], embedder_per_pose_action)
        )
        self._post_layer = post_layer

    @classmethod
    def from_their_action_encoder(
        cls,
        *,
        pose_action_tokenizer: PoseActionTokenizer,
        their_action_encoder: vnn.ActionEmbedding,
    ) -> Self:
        """Instantiate from the VIMA action encoder."""
        encoder = cls(
            pose_action_tokenizer=pose_action_tokenizer,
            embedder_per_pose_action=cast(
                dict[PoseActionType, vnn.ContinuousActionEmbedding],
                their_action_encoder._embed_dict,  # noqa: SLF001
            ),
            post_layer=cast(torch.nn.Linear, their_action_encoder._post_layer),  # noqa: SLF001
        )
        encoder.register_forward_pre_hook(_remove_zth_dimension)

        return encoder

    @property
    def num_action_tokens_per_timestep(self) -> int:
        """The number of action tokens per timestep."""
        return 1

    def forward(self, continuous_actions: dict[PoseActionType, torch.Tensor]) -> torch.Tensor:
        """Embed the continuous actions the way VIMA does it."""
        normalized_continuous_actions = self._pose_action_tokenizer.normalize_continuous_actions(
            continuous_actions
        )

        embedded_continuous_action_tokens = {
            pose_action_type: self._embedder_per_pose_action[pose_action_type](
                normalized_continuous_actions[pose_action_type]
            )
            for pose_action_type in normalized_continuous_actions
        }
        # (batch, num obs, combined dim)
        combined_embedding = torch.cat(
            [
                embedded_continuous_action_tokens[pose_action_type]
                for pose_action_type in sorted(embedded_continuous_action_tokens.keys())
            ],
            dim=-1,
        )

        fused_embedding = self._post_layer(combined_embedding)

        # Add an additional dim because each timestep consists of only a single token
        fused_embedding = fused_embedding.unsqueeze(-2)

        return fused_embedding


class TokenPerAxisActionEmbedder(ActionEncoder):
    """Embed actions to a single token per axis.

    This is so that we can autoregressively decode them.
    """

    def __init__(
        self,
        *,
        pose_action_tokenizer: PoseActionTokenizer,
        num_axes: int,
        max_num_action_bins: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self._pose_action_tokenizer = pose_action_tokenizer
        self._num_axes = num_axes
        self._max_num_action_bins = max_num_action_bins

        self.pose_action_axes_embedder = torch.nn.Parameter(
            torch.empty(
                self._num_axes, self._max_num_action_bins, embed_dim, device=None, dtype=None
            ),
            requires_grad=True,
        )
        torch.nn.init.normal_(self.pose_action_axes_embedder)

    @property
    def num_action_tokens_per_timestep(self) -> int:
        """The number of action tokens per timestep."""
        return self._num_axes

    def forward(self, continuous_actions: dict[PoseActionType, torch.Tensor]) -> torch.Tensor:
        """Embed the actions into multiple tokens, so we can autoregressively decode them."""
        discrete_actions = self._pose_action_tokenizer.convert_continuous_to_discrete(
            continuous_actions
        )
        discrete_actions_tensor = torch.cat(
            [
                discrete_actions[pose_action_type]
                for pose_action_type in sorted(discrete_actions.keys())
            ],
            dim=-1,
        )
        embedded_actions = self._embed_discrete_actions(discrete_actions_tensor)

        return embedded_actions

    def _embed_discrete_actions(self, discrete_actions_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, num_timesteps, num_axes = discrete_actions_tensor.shape

        action_indices = discrete_actions_tensor
        action_indices = rearrange(action_indices, "B T A -> (B T) A")
        action_indices = rearrange(action_indices, "BT A -> (BT A)")

        batch_indices = torch.arange(
            num_axes, device=discrete_actions_tensor.device, dtype=torch.long
        ).tile(batch_size * num_timesteps)

        embedded_actions = self.pose_action_axes_embedder[batch_indices, action_indices]

        embedded_actions = rearrange(
            embedded_actions, "(B T A) dim -> B T A dim", B=batch_size, T=num_timesteps, A=num_axes
        )
        return embedded_actions
