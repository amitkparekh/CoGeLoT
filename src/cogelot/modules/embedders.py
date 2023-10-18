import abc
from typing import Self, cast

import torch

from cogelot.modules.tokenizers.pose_action import PoseActionTokenizer
from cogelot.structures.vima import PoseActionType
from vima import nn as vnn


class ActionEmbedder(abc.ABC, torch.nn.Module):
    """Embed actions from their continuous form to something that is useful for the model."""

    @abc.abstractmethod
    def forward(self, continuous_actions: dict[PoseActionType, torch.Tensor]) -> torch.Tensor:
        """Embed the continuous actions."""
        raise NotImplementedError


class LearnedActionEmbedder(ActionEmbedder):
    """Embed actions using learned embeddings.

    Different to the way VIMA does it but is how the original Gato does it.
    """

    def __init__(
        self,
        *,
        embedder_per_pose_action: dict[PoseActionType, torch.nn.Embedding],
        output_dim: int,
    ) -> None:
        super().__init__()
        self._embedder_per_pose_action = embedder_per_pose_action
        self._output_dim = output_dim
        self._post_layer = torch.nn.LazyLinear(output_dim)

        raise NotImplementedError("This implementation is not finished yet.")

    def forward(self, discrete_action_tokens: dict[PoseActionType, torch.Tensor]) -> torch.Tensor:
        """Embed the discrete action tokens."""
        embedded_action_tokens = {
            pose_action_type: self._embedder_per_pose_action[pose_action_type](
                discrete_action_tokens[pose_action_type]
            )
            for pose_action_type in discrete_action_tokens
        }

        combined_embedding = torch.cat(
            [
                embedded_action_tokens[pose_action_type]
                for pose_action_type in sorted(embedded_action_tokens.keys())
            ],
            dim=-1,
        )

        fused_embedding = self._post_layer(combined_embedding)

        return fused_embedding


class VIMAContinuousActionEmbedder(ActionEmbedder):
    """Embed actions from their continuous form.

    First normalize the continuous actions and then encode them through an MLP.

    This follows the VIMA way of doing things.
    """

    def __init__(
        self,
        *,
        pose_action_tokenizer: PoseActionTokenizer,
        embedder_per_pose_action: dict[PoseActionType, vnn.ContinuousActionEmbedding],
        post_layer: torch.nn.Linear | torch.nn.Identity | torch.nn.LazyLinear,
    ) -> None:
        super().__init__()
        self._pose_action_tokenizer = pose_action_tokenizer
        self._embedder_per_pose_action = embedder_per_pose_action
        self._post_layer = post_layer

    @classmethod
    def from_their_action_encoder(
        cls,
        *,
        pose_action_tokenizer: PoseActionTokenizer,
        their_action_encoder: vnn.ActionEmbedding,
    ) -> Self:
        """Instantiate from the VIMA action encoder."""
        return cls(
            pose_action_tokenizer=pose_action_tokenizer,
            embedder_per_pose_action=cast(
                dict[PoseActionType, vnn.ContinuousActionEmbedding],
                their_action_encoder._embed_dict,  # noqa: SLF001
            ),
            post_layer=their_action_encoder._post_layer,  # noqa: SLF001
        )

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

        combined_embedding = torch.cat(
            [
                embedded_continuous_action_tokens[pose_action_type]
                for pose_action_type in sorted(embedded_continuous_action_tokens.keys())
            ],
            dim=-1,
        )

        fused_embedding = self._post_layer(combined_embedding)

        return fused_embedding
