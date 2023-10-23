from operator import itemgetter
from typing import ClassVar, Self, cast

import torch

from cogelot.modules.embedders import ActionEmbedder, VIMAContinuousActionEmbedder
from cogelot.modules.stitching import (
    add_observations_to_tokens_using_scatter,
    get_max_num_objects_from_encoded_observations,
    stitch_observations_with_actions,
)
from cogelot.modules.tokenizers.pose_action import (
    PoseActionTokenizer,
    create_mask_from_target_actions,
)
from cogelot.nn.decoders import TransformerDecoderProtocol
from cogelot.nn.decoders.vima import VIMADecoder
from cogelot.structures.model import RawPromptTokenType
from cogelot.structures.vima import PoseActionType
from vima import nn as vnn
from vima.nn.action_decoder.dists import MultiCategorical
from vima.policy.vima_policy import VIMAPolicy
from vima.utils import DataDict


def get_max_length_of_prompt(raw_prompts_token_type: RawPromptTokenType, max_num_objs: int) -> int:
    """Get the maximum length of the prompt.

    Refactored by ChatGPT to be more efficient and readable.
    """
    length_per_prompt = (
        sum([1 if token_type == 0 else max_num_objs for token_type in raw_prompt])
        for raw_prompt in raw_prompts_token_type
    )
    max_length = max(length_per_prompt)
    return max_length


class Policy(torch.nn.Module):
    """Common policy with compositional modules for easy swapping."""

    _views: ClassVar[list[str]] = ["front", "top"]

    def __init__(
        self,
        *,
        embed_dim: int,
        obj_encoder: vnn.ObjEncoder,
        end_effector_encoder: vnn.Embedding,
        obs_fusion_layer: torch.nn.Linear,
        action_encoder: ActionEmbedder,
        action_decoder: vnn.ActionDecoder,
        prompt_embedding: vnn.WordEmbedding,
        prompt_encoder: vnn.T5PromptEncoder,
        prompt_obj_post_layer: torch.nn.Sequential,
        transformer_decoder: TransformerDecoderProtocol,
        pose_action_tokenizer: PoseActionTokenizer,
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
        self._pose_action_tokenizer = pose_action_tokenizer

    @classmethod
    def from_their_policy(cls, their_policy: VIMAPolicy) -> Self:
        """Instantiate our policy from their policy.

        This is what we need when we are going to run from their checkpoint. The incoming policy
        has already been loaded with the correct weights so we can just rearrange the components as
        necessary.
        """
        pose_action_tokenizer = PoseActionTokenizer()
        policy = cls(
            embed_dim=their_policy.embed_dim,
            obj_encoder=their_policy.obj_encoder,
            end_effector_encoder=their_policy.end_effector_encoder,
            obs_fusion_layer=their_policy.obs_fusion_layer,
            action_encoder=VIMAContinuousActionEmbedder.from_their_action_encoder(
                pose_action_tokenizer=pose_action_tokenizer,
                their_action_encoder=their_policy.action_encoder,
            ),
            action_decoder=their_policy.action_decoder,
            prompt_embedding=their_policy.prompt_embedding,
            prompt_encoder=their_policy.t5_prompt_encoder,
            prompt_obj_post_layer=their_policy.prompt_obj_post_layer,
            transformer_decoder=VIMADecoder(their_policy.xattn_gpt),
            pose_action_tokenizer=pose_action_tokenizer,
        )
        # Take their prompt encoder post layer and replace whatever we have, otherwise it's not
        # identical. I made a mistake when originally preparing the class, however I don't know how
        # else to do it because Hydra/OmegaConf doesn't support conditionals.
        policy._prompt_encoder_post_layer = (  # noqa: SLF001
            their_policy.t5_prompt_encoder_post_layer
        )
        return policy

    def predict_action_token(
        self,
        encoded_prompt: torch.Tensor,
        encoded_prompt_mask: torch.Tensor,
        encoded_observations: torch.Tensor,
        encoded_observations_mask: torch.Tensor,
        encoded_actions: torch.Tensor | None,
        encoded_actions_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Predict the action token."""
        max_objects = get_max_num_objects_from_encoded_observations(encoded_observations)
        tokens, masks = stitch_observations_with_actions(
            encoded_observations,
            encoded_observations_mask,
            encoded_actions,
            encoded_actions_mask,
            add_observations_to_tokens_fn=add_observations_to_tokens_using_scatter,
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

        Taken from vima.policy.VIMAPolicy. Refactored with to work with batched prompts.
        """
        device = prompts[1].device
        raw_prompts_token_type, word_batch, image_batch = prompts

        batch_word_emb = self._prompt_embedding(word_batch)
        batch_image_emb = self._obj_encoder(**image_batch)
        batch_image_emb = self._prompt_obj_post_layer(batch_image_emb)

        prompt_tokens: list[torch.Tensor] = []
        prompt_masks: list[torch.Tensor] = []

        for batch_idx, raw_prompt_tokens in enumerate(raw_prompts_token_type):
            word_positions: list[int] = (
                torch.tensor(raw_prompt_tokens, device="cpu").eq(0).nonzero().flatten().tolist()
            )
            num_words = len(word_positions)
            word_embedded_per_position = batch_word_emb[batch_idx, :num_words]
            word_embedding_with_positions: list[tuple[int, torch.Tensor]] = list(
                zip(word_positions, word_embedded_per_position.split(1), strict=True)
            )
            word_masks_with_positions: list[tuple[int, torch.Tensor]] = list(
                zip(
                    word_positions,
                    torch.ones(len(word_positions), dtype=torch.bool, device=device).split(1),
                    strict=True,
                )
            )

            image_positions: list[int] = (
                torch.tensor(raw_prompt_tokens, device="cpu").eq(1).nonzero().flatten().tolist()
            )
            num_images = len(image_positions)
            embedded_images = batch_image_emb[batch_idx, :num_images]

            image_embedding_with_positions: list[tuple[int, torch.Tensor]] = list(
                zip(
                    image_positions,
                    embedded_images.reshape(-1, self.embed_dim).chunk(num_images),
                    strict=True,
                )
            )

            image_mask_with_positions = list(
                zip(
                    image_positions,
                    torch.cat(
                        list(map(itemgetter(1), sorted(image_batch[batch_idx]["mask"].items()))),
                        dim=-1,
                    )[:num_images]
                    .reshape(-1)
                    .chunk(num_images),
                    strict=True,
                )
            )

            # Merge the two lists together, and sort them by position, and just get the embeddings
            # (removing the index)
            assembled_prompt = torch.cat(
                list(
                    map(
                        itemgetter(1),
                        sorted([*word_embedding_with_positions, *image_embedding_with_positions]),
                    )
                ),
                dim=0,
            )

            assembled_mask = torch.cat(
                list(
                    map(
                        itemgetter(1),
                        sorted([*word_masks_with_positions, *image_mask_with_positions]),
                    )
                ),
                dim=0,
            )

            prompt_tokens.append(assembled_prompt)
            prompt_masks.append(assembled_mask)

        prompt_tokens_tensor = torch.nn.utils.rnn.pad_sequence(prompt_tokens, batch_first=True)
        prompt_masks_tensor = torch.nn.utils.rnn.pad_sequence(prompt_masks, batch_first=True)

        # Convert to the PyTorch-style mask, where True means it IS MASKED. The VIMA source opts
        # for the other approach, and we are going to be consistent dammit.
        prompt_masks_tensor = ~prompt_masks_tensor
        return prompt_tokens_tensor, prompt_masks_tensor

    def encode_observation_token(self, observation: DataDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an observation.

        When creating the mask, we follow PyTorch's meaning: `True` == "IS masked".
        """
        obs_objects, ee = observation["objects"], observation["ee"]

        assert isinstance(obs_objects, DataDict)
        assert isinstance(ee, torch.Tensor)

        leading_dims = ee.shape[:2]

        # Get the features for each image/obj/obs
        obs_objects = obs_objects.map_structure(
            func=lambda x: x.reshape(-1, *x.shape[3:]),
        )
        # I know the type from manual inspection
        obs_objects = cast(dict[str, dict[str, torch.Tensor]], obs_objects)
        img_feats = self._obj_encoder(**obs_objects)
        img_feats = img_feats.reshape(*leading_dims, *img_feats.shape[1:])

        # Get EE features
        ee_feats: torch.Tensor = self._end_effector_encoder(ee)
        ee_feats = ee_feats.repeat(1, 1, img_feats.shape[-2], 1)  # noqa: WPS221

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

        # Convert to the PyTorch-style mask, where True means it IS MASKED. The VIMA source opts
        # for the other approach, and we are going to be consistent dammit.
        obj_mask_tensor = ~obj_mask_tensor

        return obs_feats, obj_mask_tensor

    def encode_action_tokens(
        self,
        actions: DataDict | dict[PoseActionType, torch.Tensor],
        *,
        ignore_target_index: int = -100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the continuous actions.

        So this takes the action in their original continuous form and embeds/encodes them.
        """
        mask = create_mask_from_target_actions(actions, ignore_target_index=ignore_target_index)
        encoded_actions = self._action_encoder(actions)
        return encoded_actions, mask

    def encode_prompt(
        self, embedded_prompt: torch.Tensor, embedded_prompt_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode the prompt."""
        # Since we are using torch-style mask meaning, we need to invert the mask for the HF model
        prompt_tokens = self._prompt_encoder(
            embedded_prompt, attention_mask=~embedded_prompt_mask, batch_first=True
        )
        prompt_tokens = self._prompt_encoder_post_layer(prompt_tokens)
        return prompt_tokens

    def decode_action_token(
        self, predicted_action_tokens: torch.Tensor
    ) -> dict[PoseActionType, MultiCategorical]:
        """Decode the action token."""
        return self._action_decoder(predicted_action_tokens)

    def tokenize_continuous_actions(
        self, continuous_actions: dict[PoseActionType, torch.Tensor]
    ) -> dict[PoseActionType, torch.Tensor]:
        """Convert the continuous actions into a discrete form to work with cross-entropy."""
        return self._pose_action_tokenizer.convert_continuous_to_discrete(continuous_actions)
