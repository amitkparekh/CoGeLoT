from typing import TYPE_CHECKING

import torch
from pytest_cases import AUTO, fixture, parametrize

from cogelot.environment.vima import VIMAEnvironment
from cogelot.models import VIMALightningModule
from cogelot.modules.action_decoders import VIMAActionDecoder
from cogelot.modules.action_encoders import VIMAContinuousActionEmbedder
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
def vima_continuous_action_embedder(
    pose_action_tokenizer: PoseActionTokenizer,
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
        post_layer=torch.nn.LazyLinear(768),
    )


@fixture(scope="session")
@parametrize("add_residual_connection", [False, True], idgen=AUTO)  # pyright: ignore[reportGeneralTypeIssues]
def vima_policy(
    vima_continuous_action_embedder: VIMAContinuousActionEmbedder,
    pose_action_tokenizer: PoseActionTokenizer,
    *,
    add_residual_connection: bool,
) -> Policy:
    vima = VIMAPolicy(embed_dim=768, xf_n_layers=2, sattn_n_heads=2, xattn_n_heads=2)
    return Policy(
        embed_dim=vima.embed_dim,
        obj_encoder=vima.obj_encoder,
        end_effector_encoder=vima.end_effector_encoder,
        obs_fusion_layer=vima.obs_fusion_layer,
        action_encoder=vima_continuous_action_embedder,
        action_decoder=VIMAActionDecoder(vima.action_decoder),
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
