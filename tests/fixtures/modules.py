from pytest_cases import fixture

from cogelot.data.preprocess import InstancePreprocessor
from cogelot.models.vima import VIMALightningModule
from cogelot.modules.preprocessors.their_instance_batcher import TheirInstanceBatcher
from cogelot.modules.tokenizers import (
    EndEffectorTokenizer,
    ImageTokenizer,
    MultimodalHistoryTokenizer,
    MultimodalPromptTokenizer,
    ObservationTokenizer,
    PoseActionTokenizer,
    TextTokenizer,
)
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
def vima_policy() -> VIMAPolicy:
    return VIMAPolicy(embed_dim=768, xf_n_layers=2, sattn_n_heads=2, xattn_n_heads=2)


@fixture(scope="session")
def their_instance_batcher(vima_policy: VIMAPolicy) -> TheirInstanceBatcher:
    return TheirInstanceBatcher(vima_policy=vima_policy)


@fixture(scope="session")
def vima_lightning_module(vima_policy: VIMAPolicy) -> VIMALightningModule:
    return VIMALightningModule(vima_policy=vima_policy)
