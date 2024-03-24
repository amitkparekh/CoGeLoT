from typing import TYPE_CHECKING

import torch
from pytest_cases import fixture, parametrize_with_cases
from transformers.models.t5.configuration_t5 import T5Config

from cogelot.environment.vima import VIMAEnvironment
from cogelot.models import EvaluationLightningModule, VIMALightningModule
from cogelot.modules.action_decoders import (
    ActionDecoder,
    TokenPerAxisActionDecoder,
    VIMAActionDecoder,
)
from cogelot.modules.action_encoders import (
    ActionEncoder,
    TokenPerAxisActionEmbedder,
    VIMAContinuousActionEmbedder,
)
from cogelot.modules.instance_preprocessor import InstancePreprocessor
from cogelot.modules.policy import Policy
from cogelot.modules.text_encoders import T5PromptEncoder, T5TextEmbedder
from cogelot.modules.tokenizers import (
    EndEffectorTokenizer,
    PoseActionTokenizer,
    TextTokenizer,
)
from cogelot.nn.decoders.interfaces import (
    TransformerDecoderProtocol,
)
from cogelot.nn.decoders.torch import TorchVanillaDecoder
from cogelot.nn.decoders.vima import VIMADecoder
from vima import nn as vnn
from vima.policy import VIMAPolicy

if TYPE_CHECKING:
    from cogelot.structures.vima import PoseActionType


@fixture(scope="session")
def pretrained_model() -> str:
    return "hf-internal-testing/tiny-random-t5"


@fixture(scope="session")
def embed_dim(pretrained_model: str) -> int:
    return T5Config.from_pretrained(pretrained_model).d_model


@fixture(scope="session")
def text_tokenizer(pretrained_model: str) -> TextTokenizer:
    return TextTokenizer(pretrained_model=pretrained_model)


@fixture(scope="session")
def end_effector_tokenizer() -> EndEffectorTokenizer:
    return EndEffectorTokenizer()


@fixture(scope="session")
def pose_action_tokenizer() -> PoseActionTokenizer:
    return PoseActionTokenizer()


@fixture(scope="session")
def instance_preprocessor(
    text_tokenizer: TextTokenizer,
    end_effector_tokenizer: EndEffectorTokenizer,
) -> InstancePreprocessor:
    return InstancePreprocessor(
        text_tokenizer=text_tokenizer,
        end_effector_tokenizer=end_effector_tokenizer,
    )


@fixture(scope="session")
def prompt_embedder(pretrained_model: str) -> T5TextEmbedder:
    return T5TextEmbedder(pretrained_model)


@fixture(scope="session")
def prompt_encoder(pretrained_model: str) -> T5PromptEncoder:
    encoder = T5PromptEncoder.from_pretrained(pretrained_model, unfreeze_last_n_layers=2)
    assert isinstance(encoder, torch.nn.Module)
    assert isinstance(encoder, T5PromptEncoder)
    return encoder


@fixture(scope="session")
def object_encoder(embed_dim: int) -> vnn.ObjEncoder:
    return vnn.ObjEncoder(
        transformer_emb_dim=embed_dim,
        views=["front", "top"],
        vit_output_dim=embed_dim,
        vit_resolution=32,
        vit_patch_size=16,
        vit_width=embed_dim,
        vit_layers=4,
        vit_heads=2,
        bbox_mlp_hidden_dim=embed_dim,
        bbox_mlp_hidden_depth=2,
    )


@fixture(scope="session")
def end_effector_encoder() -> torch.nn.Embedding:
    return torch.nn.Embedding(num_embeddings=2, embedding_dim=2)


@fixture(scope="session")
def obs_fusion_layer(object_encoder: vnn.ObjEncoder, embed_dim: int) -> torch.nn.Linear:
    return torch.nn.Linear(object_encoder.output_dim + 2, embed_dim)


@fixture(scope="session")
def prompt_obj_post_layer(object_encoder: vnn.ObjEncoder, embed_dim: int) -> torch.nn.Sequential:
    return vnn.build_mlp(
        object_encoder.output_dim, hidden_dim=embed_dim, output_dim=embed_dim, hidden_depth=2
    )


class ActionEncoderDecoderCases:
    def case_single_action_token(
        self, pose_action_tokenizer: PoseActionTokenizer, embed_dim: int
    ) -> tuple[VIMAContinuousActionEmbedder, VIMAActionDecoder]:
        embedder_per_pose_action: dict[PoseActionType, vnn.ContinuousActionEmbedding] = {
            "pose0_position": vnn.ContinuousActionEmbedding(
                output_dim=256,
                input_dim=3,
                hidden_dim=256,
                hidden_depth=1,
            ),
            "pose0_rotation": vnn.ContinuousActionEmbedding(
                output_dim=256,
                input_dim=4,
                hidden_dim=256,
                hidden_depth=1,
            ),
            "pose1_position": vnn.ContinuousActionEmbedding(
                output_dim=256,
                input_dim=3,
                hidden_dim=256,
                hidden_depth=1,
            ),
            "pose1_rotation": vnn.ContinuousActionEmbedding(
                output_dim=256,
                input_dim=4,
                hidden_dim=256,
                hidden_depth=1,
            ),
        }
        encoder = VIMAContinuousActionEmbedder(
            pose_action_tokenizer=pose_action_tokenizer,
            embedder_per_pose_action=embedder_per_pose_action,
            post_layer=torch.nn.LazyLinear(embed_dim),
        )
        action_decoder = vnn.ActionDecoder(
            input_dim=embed_dim,
            action_dims={
                "pose0_position": [50, 100, 50],
                "pose0_rotation": [50] * 4,
                "pose1_position": [50, 100, 50],
                "pose1_rotation": [50] * 4,
            },
            hidden_dim=512,
            hidden_depth=2,
            activation="relu",
            norm_type=None,
            last_layer_gain=0.01,
        )
        decoder = VIMAActionDecoder(action_decoder)
        return encoder, decoder

    # @pytest.mark.skip("Only focusing on single tokens for now.")
    def case_token_per_axis(
        self, pose_action_tokenizer: PoseActionTokenizer, embed_dim: int
    ) -> tuple[TokenPerAxisActionEmbedder, TokenPerAxisActionDecoder]:
        encoder = TokenPerAxisActionEmbedder(
            pose_action_tokenizer=pose_action_tokenizer,
            num_axes=14,
            max_num_action_bins=100,
            embed_dim=embed_dim,
        )

        decoder = TokenPerAxisActionDecoder(
            input_dim=embed_dim, max_num_action_bins=100, num_action_tokens_per_timestep=14
        )
        return encoder, decoder


class TransformerDecoderCases:
    def case_vima_decoder(self, embed_dim: int) -> VIMADecoder:
        vima = VIMAPolicy(embed_dim=embed_dim, xf_n_layers=2, sattn_n_heads=2, xattn_n_heads=2)
        return VIMADecoder(vima.xattn_gpt)

    def case_torch_decoder(self, embed_dim: int) -> TorchVanillaDecoder:
        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=2,
            dim_feedforward=embed_dim,
            dropout=0,
            batch_first=True,
        )
        pos_embedder = torch.nn.Embedding(512, embed_dim)
        xattn_embedder = torch.nn.Embedding(512, embed_dim)
        return TorchVanillaDecoder(
            decoder=torch.nn.TransformerDecoder(decoder_layer, num_layers=2),
            pos_embedder=pos_embedder,
            xattn_embedder=xattn_embedder,
        )


@fixture(scope="session")
@parametrize_with_cases("action_encoder_decoder", cases=ActionEncoderDecoderCases, scope="session")
@parametrize_with_cases("transformer_decoder", cases=TransformerDecoderCases, scope="session")
def vima_policy(
    embed_dim: int,
    action_encoder_decoder: tuple[ActionEncoder, ActionDecoder],
    transformer_decoder: TransformerDecoderProtocol,
    pose_action_tokenizer: PoseActionTokenizer,
    prompt_encoder: T5PromptEncoder,
    prompt_embedder: T5TextEmbedder,
    object_encoder: vnn.ObjEncoder,
    end_effector_encoder: torch.nn.Embedding,
    obs_fusion_layer: torch.nn.Linear,
    prompt_obj_post_layer: torch.nn.Sequential,
) -> Policy:
    return Policy(
        embed_dim=embed_dim,
        obj_encoder=object_encoder,
        end_effector_encoder=end_effector_encoder,
        obs_fusion_layer=obs_fusion_layer,
        action_encoder=action_encoder_decoder[0],
        action_decoder=action_encoder_decoder[1],
        prompt_embedding=prompt_embedder,
        prompt_encoder=prompt_encoder,
        prompt_obj_post_layer=prompt_obj_post_layer,
        transformer_decoder=transformer_decoder,
        pose_action_tokenizer=pose_action_tokenizer,
    )


@fixture(scope="session")
def vima_lightning_module(vima_policy: Policy) -> VIMALightningModule:
    return VIMALightningModule(policy=vima_policy)


@fixture(scope="session")
def vima_environment() -> VIMAEnvironment:
    return VIMAEnvironment.from_config(task=1, partition=1, seed=10)


@fixture
def evaluation_module(
    instance_preprocessor: InstancePreprocessor,
    vima_lightning_module: VIMALightningModule,
    vima_environment: VIMAEnvironment,
) -> EvaluationLightningModule:
    return EvaluationLightningModule(
        environment=vima_environment,
        model=vima_lightning_module,
        instance_preprocessor=instance_preprocessor,
        max_timesteps=2,
    )
