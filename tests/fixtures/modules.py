from pytest_cases import fixture

from cogelot.data.preprocess import InstancePreprocessor
from cogelot.environment.vima import VIMAEnvironment
from cogelot.models import EvaluationLightningModule, VIMALightningModule
from cogelot.modules.policy import Policy
from cogelot.modules.tokenizers import (
    EndEffectorTokenizer,
    ImageTokenizer,
    MultimodalHistoryTokenizer,
    MultimodalPromptTokenizer,
    ObservationTokenizer,
    PoseActionTokenizer,
    TextTokenizer,
)
from cogelot.nn.decoders.vima import VIMADecoder
from cogelot.structures.common import View
from vima.policy import VIMAPolicy


@fixture(scope="session")
def text_tokenizer(pretrained_model: str) -> TextTokenizer:
    return TextTokenizer(pretrained_model=pretrained_model)


@fixture(scope="session")
def image_tokenizer() -> ImageTokenizer:
    return ImageTokenizer()


@fixture(scope="session")
def end_effector_tokenizer() -> EndEffectorTokenizer:
    return EndEffectorTokenizer()


@fixture(scope="session")
def pose_action_tokenizer() -> PoseActionTokenizer:
    return PoseActionTokenizer(
        n_discrete_x_bins=50,
        n_discrete_y_bins=100,
        n_discrete_z_bins=50,
        n_discrete_rotation_bins=50,
    )


@fixture(scope="session")
def observation_tokenizer(
    image_tokenizer: ImageTokenizer, end_effector_tokenizer: EndEffectorTokenizer
) -> ObservationTokenizer:
    return ObservationTokenizer(
        image_tokenizer=image_tokenizer, end_effector_tokenizer=end_effector_tokenizer
    )


@fixture(scope="session")
def multimodal_prompt_tokenizer(
    text_tokenizer: TextTokenizer, image_tokenizer: ImageTokenizer
) -> MultimodalPromptTokenizer:
    return MultimodalPromptTokenizer(
        text_tokenizer=text_tokenizer,
        image_tokenizer=image_tokenizer,
        views=[View.front, View.top],
    )


@fixture(scope="session")
def multimodal_history_tokenizer(
    observation_tokenizer: ObservationTokenizer,
    pose_action_tokenizer: PoseActionTokenizer,
) -> MultimodalHistoryTokenizer:
    return MultimodalHistoryTokenizer(
        observation_tokenizer=observation_tokenizer,
        pose_action_tokenizer=pose_action_tokenizer,
    )


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


@fixture(scope="session")
def evaluation_module(
    instance_preprocessor: InstancePreprocessor,
    vima_lightning_module: VIMALightningModule,
    vima_environment: VIMAEnvironment,
) -> EvaluationLightningModule:
    return EvaluationLightningModule(
        environment=vima_environment,
        model=vima_lightning_module,
        preprocessor=instance_preprocessor,
    )
