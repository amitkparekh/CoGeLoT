from typing import ClassVar, Self, cast

import torch
from transformers.models.t5.modeling_t5 import T5EncoderModel

from cogelot.modules.action_decoders import ActionDecoder, VIMAActionDecoder
from cogelot.modules.action_encoders import ActionEncoder, VIMAContinuousActionEmbedder
from cogelot.modules.prompt_assembler import assemble_multimodal_prompt
from cogelot.modules.stitching import (
    add_encoding_to_tokens_using_scatter,
    stitch_observations_with_actions,
)
from cogelot.modules.text_encoders import T5TextEmbedder
from cogelot.modules.tokenizers.pose_action import (
    PoseActionTokenizer,
    create_mask_from_target_actions,
)
from cogelot.nn.decoders import TransformerDecoderGreedyGenerateWrapper, TransformerDecoderProtocol
from cogelot.nn.decoders.vima import VIMADecoder
from cogelot.structures.model import RawPromptTokenType
from cogelot.structures.vima import PoseActionType
from vima import nn as vnn
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
        end_effector_encoder: vnn.Embedding | torch.nn.Embedding,
        obs_fusion_layer: torch.nn.Linear,
        action_encoder: ActionEncoder,
        action_decoder: ActionDecoder,
        prompt_embedding: T5TextEmbedder,
        prompt_encoder: T5EncoderModel,
        prompt_obj_post_layer: torch.nn.Sequential,
        transformer_decoder: TransformerDecoderProtocol,
        pose_action_tokenizer: PoseActionTokenizer,
        add_residual_connection_to_prompt_visual_features: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        if (
            action_encoder.num_action_tokens_per_timestep
            != action_decoder.num_action_tokens_per_timestep
        ):
            raise AssertionError(
                "The number of action tokens per timestep expected by both the encoder and decoder are not the same."
            )

        self._obj_encoder = obj_encoder
        self._end_effector_encoder = end_effector_encoder
        self._obs_fusion_layer = obs_fusion_layer
        self._action_encoder = action_encoder
        self._action_decoder = action_decoder
        self._prompt_embedding = prompt_embedding
        self._prompt_encoder = prompt_encoder
        self._prompt_encoder_post_layer = (
            torch.nn.Identity()
            if embed_dim == self._prompt_encoder.config.d_model
            else torch.nn.Linear(self._prompt_encoder.config.d_model, embed_dim, bias=False)
        )
        self._prompt_obj_post_layer = prompt_obj_post_layer
        self._transformer_decoder = transformer_decoder
        self.pose_action_tokenizer = pose_action_tokenizer

        self._add_residual_connection_to_prompt_visual_features = (
            add_residual_connection_to_prompt_visual_features
        )

    @property
    def prompt_embedding(self) -> T5TextEmbedder:
        """The text embedding.

        I could make the private a public attribute, but that means I then need to do model surgery
        on the checkpoints for the weights and I don't want to be dirty.
        """
        return self._prompt_embedding

    @classmethod
    def from_their_policy(cls, their_policy: VIMAPolicy) -> Self:
        """Instantiate our policy from their policy.

        This is what we need when we are going to run from their checkpoint. The incoming policy
        has already been loaded with the correct weights so we can just rearrange the components as
        necessary.
        """
        pose_action_tokenizer = PoseActionTokenizer(remove_z_position_dim=True)
        policy = cls(
            embed_dim=their_policy.embed_dim,
            obj_encoder=their_policy.obj_encoder,
            end_effector_encoder=their_policy.end_effector_encoder,
            obs_fusion_layer=their_policy.obs_fusion_layer,
            action_encoder=VIMAContinuousActionEmbedder.from_their_action_encoder(
                pose_action_tokenizer=pose_action_tokenizer,
                their_action_encoder=their_policy.action_encoder,
            ),
            action_decoder=VIMAActionDecoder(their_policy.action_decoder),
            prompt_embedding=cast(T5TextEmbedder, their_policy.prompt_embedding),
            prompt_encoder=their_policy.t5_prompt_encoder.t5,  # pyright: ignore[reportGeneralTypeIssues]
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

    @property
    def num_action_tokens_per_timestep(self) -> int:
        """The number of action tokens per timestep."""
        return self._action_encoder.num_action_tokens_per_timestep

    def predict_action_logits(
        self,
        encoded_prompt: torch.Tensor,
        encoded_prompt_mask: torch.Tensor,
        encoded_observations: torch.Tensor,
        encoded_observations_mask: torch.Tensor,
        encoded_actions: torch.Tensor | None,
        encoded_actions_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Predict the action token."""
        tokens, masks = stitch_observations_with_actions(
            encoded_observations,
            encoded_observations_mask,
            encoded_actions,
            encoded_actions_mask,
            add_encoding_to_tokens_fn=add_encoding_to_tokens_using_scatter,
            num_action_tokens_per_timestep=self.num_action_tokens_per_timestep,
        )

        transformer_output = self._transformer_decoder(
            tgt=tokens,
            tgt_key_padding_mask=masks,
            memory=encoded_prompt,
            memory_key_padding_mask=encoded_prompt_mask,
        )
        predicted_actions = self.decode_action_logits(
            transformer_output, max_num_objects=encoded_observations.size(-2)
        )
        return predicted_actions

    def embed_multimodal_prompt(
        self, prompts: tuple[RawPromptTokenType, torch.Tensor, DataDict]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assembed, embed, and encode the prompt."""
        device = prompts[1].device
        raw_prompts_token_type, word_batch, image_batch = prompts

        embedded_words = self._prompt_embedding(word_batch)
        embedded_images = self._obj_encoder(**image_batch)
        embedded_images = self._prompt_obj_post_layer(embedded_images)

        embedded_prompt, embedded_prompt_mask = assemble_multimodal_prompt(
            embedded_text=embedded_words,
            embedded_visuals=embedded_images,
            original_visuals=image_batch,
            raw_prompts_token_type=raw_prompts_token_type,
            embed_dim=self.embed_dim,
            device=device,
        )

        if self._add_residual_connection_to_prompt_visual_features:
            raise NotImplementedError("Not implemented this yet")

        return embedded_prompt, embedded_prompt_mask

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
        # Shape: (batch size, timesteps, tokens per timestep, embed_dim)
        encoded_actions = self._action_encoder(actions)

        # Shape: (batch size, timesteps)
        mask = create_mask_from_target_actions(actions, ignore_target_index=ignore_target_index)
        # Shape: (batch size, timesteps, tokens per timestep)
        encoded_actions_mask = mask.unsqueeze(-1).expand(encoded_actions.shape[: mask.ndim + 1])

        return encoded_actions, encoded_actions_mask

    def encode_prompt(
        self,
        embedded_prompt: torch.Tensor,
        embedded_prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the prompt."""
        # Since we are using torch-style mask meaning, we need to invert the mask for the HF model
        prompt_tokens = self._prompt_encoder(
            inputs_embeds=embedded_prompt, attention_mask=~embedded_prompt_mask
        ).last_hidden_state
        prompt_tokens = self._prompt_encoder_post_layer(prompt_tokens)

        return prompt_tokens

    def decode_action_logits(
        self, transformer_output: torch.Tensor, max_num_objects: int
    ) -> torch.Tensor:
        """Decode the action token."""
        return self._action_decoder(transformer_output, max_num_objects=max_num_objects)

    def tokenize_continuous_actions(
        self, continuous_actions: dict[PoseActionType, torch.Tensor]
    ) -> dict[PoseActionType, torch.Tensor]:
        """Convert the continuous actions into a discrete form to work with cross-entropy."""
        return self.pose_action_tokenizer.convert_continuous_to_discrete(continuous_actions)

    def prepare_policy_for_greedy_generation(self) -> None:
        """Prepare the policy for greedy generation."""
        if self.training:
            raise AssertionError(
                "The policy is in training mode, so we should not prepare it for greedy generation since that is just inefficient."
            )
        self._transformer_decoder = TransformerDecoderGreedyGenerateWrapper(
            self._transformer_decoder,
            num_tokens_to_generate_per_timestep=self.num_action_tokens_per_timestep,
        )
