from typing import ClassVar, cast

import torch
from einops import rearrange

from cogelot.nn.decoders import TransformerDecoderProtocol
from cogelot.structures.model import RawPromptTokenType
from cogelot.structures.vima import PoseActionType
from vima import nn as vnn
from vima.nn.action_decoder.dists import MultiCategorical
from vima.utils import DataDict


def get_max_length_of_prompt(raw_prompts_token_type: RawPromptTokenType, max_num_objs: int) -> int:
    """Get the maximum length of the prompt.

    Refactored by ChatGPT to be more efficient and readable.
    """
    max_length = 0

    for raw_prompt in raw_prompts_token_type:
        current_length = sum([1 if token_type == 0 else max_num_objs for token_type in raw_prompt])
        max_length = max(max_length, current_length)

    return max_length


class Policy(torch.nn.Module):
    """Common policy with compositional modules for easy swapping."""

    n_discrete_x_bins: int = 50
    n_discrete_y_bins: int = 100
    n_discrete_z_bins: int = 50
    n_discrete_rot_bins: int = 50
    _views: ClassVar[list[str]] = ["front", "top"]

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
        transformer_decoder: TransformerDecoderProtocol,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self._obj_encoder = obj_encoder
        self._end_effector_encoder = end_effector_encoder
        self._obs_fusion_layer = obs_fusion_layer
        self._action_encoder = action_encoder
        self._action_decoder = action_decoder
        self._prompt_embedding = prompt_embedding
        self._prompt_encoder = prompt_encoder
        self._prompt_encoder_post_layer = (
            torch.nn.Identity()
            if embed_dim == self._prompt_encoder.output_dim
            else torch.nn.Linear(self._prompt_encoder.output_dim, embed_dim, bias=False)
        )
        self._prompt_obj_post_layer = prompt_obj_post_layer
        self._transformer_decoder = transformer_decoder

    def predict_action_token(
        self,
        encoded_prompt: torch.Tensor,
        encoded_prompt_mask: torch.Tensor,
        embedded_observations: torch.Tensor,
        embedded_observations_mask: torch.Tensor,
        embedded_actions: torch.Tensor | None,
    ) -> torch.Tensor:
        """Predict the action token."""
        max_objects = self._get_max_num_objects(embedded_observations=embedded_observations)
        tokens, masks = self._stitch_observations_with_actions(
            embedded_observations, embedded_observations_mask, embedded_actions
        )

        transformer_output = self._transformer_decoder.forward(
            tgt=tokens,
            tgt_key_padding_mask=masks,
            memory=encoded_prompt,
            memory_key_padding_mask=encoded_prompt_mask,
        )
        predicted_action_tokens = transformer_output[:, max_objects - 1 :: max_objects + 1]
        return predicted_action_tokens

    def assemble_prompt(  # noqa: WPS210
        self, prompts: tuple[RawPromptTokenType, torch.Tensor, DataDict]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble and embed the prompt.

        Taken from vima.policy.VIMAPolicy. Refactored with ChatGPT to be more efficient and
        readable.
        """
        raw_prompts_token_type, word_batch, image_batch = prompts
        batch_word_emb = self._prompt_embedding(word_batch)
        batch_image_emb = self._obj_encoder(**image_batch)
        batch_image_emb = self._prompt_obj_post_layer(batch_image_emb)

        max_num_objs: int = batch_image_emb.shape[-2]
        max_prompt_length = get_max_length_of_prompt(raw_prompts_token_type, max_num_objs)

        prompt_tokens: list[torch.Tensor] = []
        prompt_masks: list[torch.Tensor] = []
        word_idx, img_idx = 0, 0

        for raw_prompt_tokens in raw_prompts_token_type:
            assembled_prompt: list[torch.Tensor] = []
            assembled_mask: list[bool] = []

            for raw_prompt_token in raw_prompt_tokens:
                # If 0, means that the token is a word
                if raw_prompt_token == 0:
                    assembled_prompt.append(batch_word_emb[word_idx])
                    assembled_mask.append(True)
                    word_idx += 1

                # If 1, means that the token is an image
                if raw_prompt_token == 1:
                    obj_masks: list[bool] = [
                        image_batch["mask"][view][  # pyright: ignore[reportGeneralTypeIssues]
                            img_idx
                        ]
                        for view in sorted(self._views)
                    ]
                    obj_embeddings = batch_image_emb[img_idx][:max_num_objs]
                    assembled_prompt.extend(obj_embeddings)
                    assembled_mask.extend(obj_masks[:max_num_objs])
                    img_idx += 1

            additional_padding_for_prompt = max_prompt_length - len(assembled_prompt)

            padding = torch.zeros(
                size=(additional_padding_for_prompt, assembled_prompt[0].shape[0]),
                dtype=torch.float32,
                device=assembled_prompt[0].device,
            )
            token_tensor_for_prompt = torch.cat(
                tensors=[torch.stack(assembled_prompt), padding], dim=0
            )
            prompt_tokens.append(token_tensor_for_prompt)

            prompt_masks.append(
                torch.cat(
                    [
                        torch.tensor(
                            assembled_mask,
                            dtype=torch.bool,
                            device=token_tensor_for_prompt.device,
                        ),
                        torch.zeros(
                            additional_padding_for_prompt,
                            dtype=torch.bool,
                            device=token_tensor_for_prompt.device,
                        ),
                    ],
                    dim=0,
                )
            )

        prompt_tokens_tensor = torch.stack(prompt_tokens, dim=0)
        prompt_masks_tensor = torch.stack(prompt_masks, dim=0)
        return prompt_tokens_tensor, prompt_masks_tensor

    def embed_observation_token(self, observation: DataDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed an observation."""
        obs_objects, ee = observation["objects"], observation["ee"]

        assert isinstance(obs_objects, DataDict)
        assert isinstance(ee, torch.Tensor)

        leading_dims = ee.shape[:2]

        # Get the features for each image/obj/obs
        obs_objects = obs_objects.map_structure(
            func=lambda x: x.reshape(-1, *x.shape[2:]),
        )
        # I know the type from manual inspection
        obs_objects = cast(dict[str, dict[str, torch.Tensor]], obs_objects)
        img_feats = self._obj_encoder(**obs_objects)
        img_feats = img_feats.reshape(*leading_dims, *img_feats.shape[1:])

        # Get EE features
        ee_feats: torch.Tensor = self._end_effector_encoder(ee)
        ee_feats = ee_feats.unsqueeze(2).repeat(1, 1, img_feats.shape[-2], 1)  # noqa: WPS221

        # Create obs features
        obs_feats: torch.Tensor = self._obs_fusion_layer(torch.cat([img_feats, ee_feats], dim=-1))

        # Create mask for obs
        obj_mask = {
            obj_key: obs_objects["mask"][obj_key].reshape(*leading_dims, -1)
            for obj_key in obs_objects["mask"]
        }
        obj_mask_tensor = torch.cat(
            [obj_mask[view] for view in sorted(self._views)],
            dim=-1,
        )

        return obs_feats, obj_mask_tensor

    def embed_action_token(self, actions: DataDict) -> torch.Tensor:
        """Embed the actions into a tensor."""
        return self._action_encoder(
            self.de_discretize_actions(cast(dict[PoseActionType, torch.Tensor], actions))
        )

    def encode_prompt(
        self, embedded_prompt: torch.Tensor, embedded_prompt_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode the prompt."""
        prompt_tokens = self._prompt_encoder(
            embedded_prompt, attention_mask=embedded_prompt_mask, batch_first=True
        )
        prompt_tokens = self._prompt_encoder_post_layer(prompt_tokens)
        return prompt_tokens

    def decode_action_token(
        self, predicted_action_tokens: torch.Tensor
    ) -> dict[PoseActionType, MultiCategorical]:
        """Decode the action token."""
        return self._action_decoder(predicted_action_tokens)

    def discretize_action(
        self, action: dict[PoseActionType, torch.Tensor]
    ) -> dict[PoseActionType, torch.Tensor]:
        """Discretize the action."""
        device = action["pose0_position"].device
        boundary_x = torch.linspace(start=0, end=1, steps=self.n_discrete_x_bins, device=device)
        boundary_y = torch.linspace(start=0, end=1, steps=self.n_discrete_y_bins, device=device)
        boundary_rot = torch.linspace(
            start=0, end=1, steps=self.n_discrete_rot_bins, device=device
        )

        action["pose0_position"][..., 0] = torch.bucketize(
            action["pose0_position"][..., 0].contiguous(), boundary_x
        )
        action["pose0_position"][..., 1] = torch.bucketize(
            action["pose0_position"][..., 1].contiguous(), boundary_y
        )
        action["pose0_rotation"] = torch.bucketize(
            action["pose0_rotation"].contiguous(), boundary_rot
        )

        action["pose1_position"][..., 0] = torch.bucketize(
            action["pose1_position"][..., 0].contiguous(), boundary_x
        )
        action["pose1_position"][..., 1] = torch.bucketize(
            action["pose1_position"][..., 1].contiguous(), boundary_y
        )
        action["pose1_rotation"] = torch.bucketize(
            action["pose1_rotation"].contiguous(), boundary_rot
        )
        action = {k: v.long() for k, v in action.items()}
        return action

    def de_discretize_actions(
        self, actions: dict[PoseActionType, torch.Tensor]
    ) -> dict[PoseActionType, torch.Tensor]:
        """De-discretize the actions."""
        actions = {k: v.float() for k, v in actions.items()}
        actions["pose0_position"][..., 0] = (
            actions["pose0_position"][..., 0] / self.n_discrete_x_bins
        )
        actions["pose0_position"][..., 1] = (
            actions["pose0_position"][..., 1] / self.n_discrete_y_bins
        )
        actions["pose0_rotation"] = actions["pose0_rotation"] / self.n_discrete_rot_bins

        actions["pose1_position"][..., 0] = (
            actions["pose1_position"][..., 0] / self.n_discrete_x_bins
        )
        actions["pose1_position"][..., 1] = (
            actions["pose1_position"][..., 1] / self.n_discrete_y_bins
        )
        actions["pose1_rotation"] = actions["pose1_rotation"] / self.n_discrete_rot_bins
        return actions

    def _stitch_observations_with_actions(
        self,
        embedded_observations: torch.Tensor,
        embedded_observations_mask: torch.Tensor,
        embedded_actions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stitch the observations together with actions for decoder input."""
        batch_size, observation_seq_len = embedded_observations.shape[:2]
        actions_seq_len = 0 if embedded_actions is None else embedded_actions.shape[1]
        max_objects = self._get_max_num_objects(embedded_observations=embedded_observations)
        total_seq_len = observation_seq_len * max_objects + actions_seq_len

        # Rearrange the tensors to be in the right structure
        # embedded_observations = rearrange(embedded_observations, "L B Q E -> B L Q E")
        embedded_observations = rearrange(embedded_observations, "B L Q E -> B (L Q) E")
        embedded_observations = rearrange(embedded_observations, "B L E -> L B E")

        # embedded_observations_mask = rearrange(embedded_observations_mask, "L B Q -> B L Q")
        embedded_observations_mask = rearrange(embedded_observations_mask, "B L Q -> B (L Q)")
        embedded_observations_mask = rearrange(embedded_observations_mask, "B L -> L B")

        # Create tensors which will we will use to put the various tokens into
        tokens = torch.empty(
            total_seq_len,
            batch_size,
            self.embed_dim,
            dtype=torch.float32,
            device=embedded_observations.device,
        )
        masks = torch.ones(
            total_seq_len, batch_size, dtype=torch.bool, device=embedded_observations.device
        )

        # Fill in the tokens and masks properly
        for obj_idx in range(max_objects):
            tokens[obj_idx :: max_objects + 1] = embedded_observations[  # noqa: WPS362
                obj_idx::max_objects
            ]
            masks[obj_idx :: max_objects + 1] = embedded_observations_mask[  # noqa: WPS362
                obj_idx::max_objects
            ]

        if embedded_actions is not None:
            tokens[max_objects :: max_objects + 1] = embedded_actions.transpose(  # noqa: WPS362
                0, 1
            )

        # Put the batch first
        tokens = rearrange(tokens, "L B E -> B L E")
        masks = rearrange(masks, "L B -> B L")

        return tokens, masks

    def _get_max_num_objects(self, *, embedded_observations: torch.Tensor) -> int:
        """Get the max number of objects."""
        return embedded_observations.shape[-2]
