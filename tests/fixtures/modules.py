from typing import TYPE_CHECKING

import pytest
import torch
from pytest_cases import fixture, param_fixture, parametrize_with_cases

from cogelot.environment.vima import VIMAEnvironment
from cogelot.models import VIMALightningModule
from cogelot.modules.action_decoders import VIMAActionDecoder
from cogelot.modules.action_encoders import (
    ActionEncoder,
    TokenPerAxisActionEmbedder,
    VIMAContinuousActionEmbedder,
)
from cogelot.modules.instance_preprocessor import InstancePreprocessor
from cogelot.modules.policy import Policy
from cogelot.modules.tokenizers import (
    EndEffectorTokenizer,
    PoseActionTokenizer,
    TextTokenizer,
)
from cogelot.nn.decoders.vima import VIMADecoder
from vima import nn as vnn
from vima.policy import VIMAPolicy

if TYPE_CHECKING:
    from cogelot.structures.vima import PoseActionType

add_residual_connection = param_fixture(
    "add_residual_connection",
    [False, pytest.param(True, marks=pytest.mark.skip())],
    ids=["without_residual_connection", "with_residual_connection"],
    scope="session",
)


@fixture(scope="session")
def pretrained_model() -> str:
    return "hf-internal-testing/tiny-random-t5"


@fixture(scope="session")
def embed_dim() -> int:
    return 768


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


class ActionEncoderCases:
    def case_vima_continuous_action_embedder(
        self,
        pose_action_tokenizer: PoseActionTokenizer,
        embed_dim: int,
    ) -> VIMAContinuousActionEmbedder:
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
        return VIMAContinuousActionEmbedder(
            pose_action_tokenizer=pose_action_tokenizer,
            embedder_per_pose_action=embedder_per_pose_action,
            post_layer=torch.nn.LazyLinear(embed_dim),
        )

    def case_token_per_axis_action_embedder(
        self,
        pose_action_tokenizer: PoseActionTokenizer,
        embed_dim: int,
    ) -> TokenPerAxisActionEmbedder:
        return TokenPerAxisActionEmbedder(
            pose_action_tokenizer=pose_action_tokenizer,
            num_axes=14,
            max_num_action_bins=100,
            embed_dim=embed_dim,
        )


@fixture(scope="session")
def vima_action_decoder(embed_dim: int) -> VIMAActionDecoder:
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
    return VIMAActionDecoder(action_decoder)


@fixture(scope="session")
@parametrize_with_cases("action_encoder", cases=ActionEncoderCases, scope="session")
def vima_policy(
    action_encoder: ActionEncoder,
    vima_action_decoder: VIMAActionDecoder,
    pose_action_tokenizer: PoseActionTokenizer,
    embed_dim: int,
    add_residual_connection: bool,  # noqa: FBT001
) -> Policy:
    vima = VIMAPolicy(embed_dim=embed_dim, xf_n_layers=2, sattn_n_heads=2, xattn_n_heads=2)
    return Policy(
        embed_dim=vima.embed_dim,
        obj_encoder=vima.obj_encoder,
        end_effector_encoder=vima.end_effector_encoder,
        obs_fusion_layer=vima.obs_fusion_layer,
        action_encoder=action_encoder,
        action_decoder=vima_action_decoder,
        prompt_embedding=vima.prompt_embedding,
        prompt_encoder=vima.t5_prompt_encoder,
        prompt_obj_post_layer=vima.prompt_obj_post_layer,
        transformer_decoder=VIMADecoder(vima.xattn_gpt),
        pose_action_tokenizer=pose_action_tokenizer,
        add_residual_connection_to_prompt_visual_features=add_residual_connection,
    )


@fixture(scope="session")
def vima_lightning_module(vima_policy: Policy) -> VIMALightningModule:
    return VIMALightningModule(policy=vima_policy)


@fixture(scope="session")
def vima_environment() -> VIMAEnvironment:
    return VIMAEnvironment.from_config(task=1, partition=1, seed=10)


# @fixture(scope="session")
# def evaluation_module(
#     instance_preprocessor: InstancePreprocessor,
#     vima_lightning_module: VIMALightningModule,
#     vima_environment: VIMAEnvironment,
# ) -> EvaluationLightningModule:
#     return EvaluationLightningModule(
#         environment=vima_environment,
#         model=vima_lightning_module,
#         preprocessor=instance_preprocessor,
#     )
