from pytest_cases import fixture

from cogelot.environment.vima import VIMAEnvironment
from cogelot.models import VIMALightningModule
from cogelot.modules.instance_preprocessor import InstancePreprocessor
from cogelot.modules.policy import Policy
from cogelot.modules.tokenizers import (
    EndEffectorTokenizer,
    PoseActionTokenizer,
    TextTokenizer,
)
from cogelot.nn.decoders.vima import VIMADecoder
from vima.policy import VIMAPolicy


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
    pose_action_tokenizer: PoseActionTokenizer,
) -> InstancePreprocessor:
    return InstancePreprocessor(
        text_tokenizer=text_tokenizer,
        end_effector_tokenizer=end_effector_tokenizer,
        pose_action_tokenizer=pose_action_tokenizer,
    )


@fixture(scope="session")
def vima_policy() -> Policy:
    vima = VIMAPolicy(embed_dim=768, xf_n_layers=2, sattn_n_heads=2, xattn_n_heads=2)
    return Policy(
        embed_dim=vima.embed_dim,
        obj_encoder=vima.obj_encoder,
        end_effector_encoder=vima.end_effector_encoder,
        obs_fusion_layer=vima.obs_fusion_layer,
        action_encoder=vima.action_encoder,
        action_decoder=vima.action_decoder,
        prompt_embedding=vima.prompt_embedding,
        prompt_encoder=vima.t5_prompt_encoder,
        prompt_obj_post_layer=vima.prompt_obj_post_layer,
        transformer_decoder=VIMADecoder(vima.xattn_gpt),
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
