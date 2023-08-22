from operator import itemgetter
from typing import ClassVar, Self, cast

import torch
from einops import rearrange

from cogelot.modules.tokenizers.pose_action import PoseActionTokenizer
from cogelot.nn.decoders import TransformerDecoderProtocol
from cogelot.nn.decoders.vima import VIMADecoder
from cogelot.structures.model import RawPromptTokenType
from cogelot.structures.vima import (
    PoseActionType,
)
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


def get_max_num_objects_from_embedded_observations(embedded_observations: torch.Tensor) -> int:
    """Get the maximum number of objects from the embedded observations."""
    return embedded_observations.shape[-2]


def stitch_observations_with_actions(  # noqa: WPS210
    embedded_observations: torch.Tensor,
    embedded_observations_mask: torch.Tensor,
    embedded_actions: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stitch the observations together with actions for decoder input."""
    embed_dim = embedded_observations.size(-1)
    batch_size, observation_seq_len = embedded_observations.shape[:2]
    actions_seq_len = 0 if embedded_actions is None else embedded_actions.shape[1]

    if observation_seq_len not in {actions_seq_len, actions_seq_len + 1}:
        raise AssertionError(
            "The number of observations must be equal to or one more than the number of actions"
        )

    max_objects = get_max_num_objects_from_embedded_observations(embedded_observations)
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
        embed_dim,
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
        tokens[max_objects :: max_objects + 1] = embedded_actions.transpose(0, 1)  # noqa: WPS362

    # Put the batch first
    tokens = rearrange(tokens, "L B E -> B L E")
    masks = rearrange(masks, "L B -> B L")

    return tokens, masks


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

        self._pose_action_tokenizer = PoseActionTokenizer()

    @classmethod
    def from_their_policy(cls, their_policy: VIMAPolicy) -> Self:
        """Instantiate our policy from their policy.

        This is what we need when we are going to run from their checkpoint.
        """
        policy = cls(
            embed_dim=their_policy.embed_dim,
            obj_encoder=their_policy.obj_encoder,
            end_effector_encoder=their_policy.end_effector_encoder,
            obs_fusion_layer=their_policy.obs_fusion_layer,
            action_encoder=their_policy.action_encoder,
            action_decoder=their_policy.action_decoder,
            prompt_embedding=their_policy.prompt_embedding,
            prompt_encoder=their_policy.t5_prompt_encoder,
            prompt_obj_post_layer=their_policy.prompt_obj_post_layer,
            transformer_decoder=VIMADecoder(their_policy.xattn_gpt),
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
        embedded_observations: torch.Tensor,
        embedded_observations_mask: torch.Tensor,
        embedded_actions: torch.Tensor | None,
    ) -> torch.Tensor:
        """Predict the action token."""
        max_objects = get_max_num_objects_from_embedded_observations(embedded_observations)
        tokens, masks = stitch_observations_with_actions(
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
                torch.tensor(raw_prompt_tokens).eq(0).nonzero().flatten().tolist()
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
                torch.tensor(raw_prompt_tokens, device=device).eq(1).nonzero().flatten().tolist()
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
        return prompt_tokens_tensor, prompt_masks_tensor

    def embed_observation_token(self, observation: DataDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed an observation."""
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

        return obs_feats, obj_mask_tensor

    def embed_action_token(
        self, actions: DataDict | dict[PoseActionType, torch.Tensor]
    ) -> torch.Tensor:
        """Embed the actions into a tensor.

        So this takes the action tokens, and converts them back into a continuous form, and then
        provides that straight into the encoder, which are just MLPs that convert from the list of
        2/4 coordinates (given the pose action type), into a N-dimensional tensor.

        I don't understand why they did this, but okay.
        """
        embedded_actions = self._action_encoder(
            self._pose_action_tokenizer.convert_discrete_to_continuous(
                cast(dict[PoseActionType, torch.Tensor], actions),
            )
        )
        return embedded_actions

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
