import torch
from einops import rearrange

from cogelot.modules.policy.policy import Policy
from vima import nn as vnn


class VIMAPolicy(Policy):
    """Put their policy in our interface.

    Refactored version of their policy. Is functionally, and essentially, identical to their
    policy.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        obj_encoder: vnn.ObjEncoder,
        end_effector_encoder: vnn.Embedding,
        obs_fusion_layer: torch.nn.Linear,
        action_encoder: vnn.ActionEmbedding,
        action_decoder: vnn.ActionDecoder,
        prompt_embedding: vnn.WordEmbedding,
        prompt_encoder: vnn.T5PromptEncoder,
        prompt_obj_post_layer: torch.nn.Sequential,
        xattn_gpt: vnn.XAttnGPT,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            obj_encoder=obj_encoder,
            end_effector_encoder=end_effector_encoder,
            obs_fusion_layer=obs_fusion_layer,
            action_encoder=action_encoder,
            action_decoder=action_decoder,
            prompt_embedding=prompt_embedding,
            prompt_encoder=prompt_encoder,
            prompt_obj_post_layer=prompt_obj_post_layer,
        )

        self._xattn_gpt = xattn_gpt

    def predict_action_token(
        self,
        encoded_prompt: torch.Tensor,
        encoded_prompt_mask: torch.Tensor,
        embedded_observations: torch.Tensor,
        embedded_observations_mask: torch.Tensor,
        embedded_actions: torch.Tensor | None,
    ) -> torch.Tensor:
        """Predict the next action."""
        return self._their_forward(
            obs_token=embedded_observations,
            obs_mask=embedded_observations_mask,
            prompt_token=encoded_prompt,
            prompt_token_mask=encoded_prompt_mask,
            action_token=embedded_actions,
        )

    def _their_forward(  # noqa: WPS210
        self,
        obs_token: torch.Tensor,
        obs_mask: torch.Tensor,
        action_token: torch.Tensor | None,
        prompt_token: torch.Tensor,
        prompt_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Rearrange the tensors to be in the right structure
        # obs_token = rearrange(obs_token, "L B Q E -> B L Q E")
        obs_token = rearrange(obs_token, "B L Q E -> B (L Q) E")
        obs_token = rearrange(obs_token, "B L E -> L B E")

        # obs_mask = rearrange(obs_mask, "L B Q -> B L Q")
        obs_mask = rearrange(obs_mask, "B L Q -> B (L Q)")
        obs_mask = rearrange(obs_mask, "B L -> L B")

        # Create tensors which will we will use to put the various tokens into
        batch_size, observation_seq_len = obs_token.shape[:2]
        actions_seq_len = 0 if action_token is None else action_token.shape[1]
        max_objects = obs_token.shape[-2]
        total_seq_len = observation_seq_len * max_objects + actions_seq_len

        tokens = torch.empty(
            total_seq_len, batch_size, self.embed_dim, dtype=torch.float32, device=obs_token.device
        )
        masks = torch.ones(total_seq_len, batch_size, dtype=torch.bool, device=obs_token.device)

        # Fill in the tokens and masks properly
        for obj_idx in range(max_objects):
            tokens[obj_idx :: max_objects + 1] = obs_token[obj_idx::max_objects]  # noqa: WPS362
            masks[obj_idx :: max_objects + 1] = obs_mask[obj_idx::max_objects]  # noqa: WPS362

        if action_token is not None:
            tokens[max_objects :: max_objects + 1] = action_token.transpose(0, 1)  # noqa: WPS362

        position_ids = torch.cumsum(masks, dim=0) - 1
        position_ids = position_ids.long()
        prompt_position_ids = torch.cumsum(prompt_token_mask, dim=1) - 1

        tokens_out = self._xattn_gpt(
            obs_action_tokens=tokens,
            prompt_tokens=prompt_token.transpose(0, 1),
            prompt_mask=prompt_token_mask,
            obs_action_masks=masks.transpose(0, 1),
            obs_action_position_ids=position_ids.transpose(0, 1),
            prompt_position_ids=prompt_position_ids,
        )

        predicted_action_tokens = tokens_out[max_objects - 1 :: max_objects + 1]
        return predicted_action_tokens
