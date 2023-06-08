from pytest_cases import fixture

from cogelot.data.structures import View
from cogelot.modules.tokenizers import (
    EndEffectorTokenizer,
    ImageTokenizer,
    MultimodalPromptTokenizer,
    ObservationTokenizer,
    TextTokenizer,
)
from vima.policy import VIMAPolicy


@fixture(scope="session")
def text_tokenizer(pretrained_model: str) -> TextTokenizer:
    return TextTokenizer(pretrained_model=pretrained_model)


@fixture(scope="session")
def image_tokenizer() -> ImageTokenizer:
    return ImageTokenizer()


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
def end_effector_tokenizer() -> EndEffectorTokenizer:
    return EndEffectorTokenizer()


@fixture(scope="session")
def observation_tokenizer(
    image_tokenizer: ImageTokenizer, end_effector_tokenizer: EndEffectorTokenizer
) -> ObservationTokenizer:
    return ObservationTokenizer(
        image_tokenizer=image_tokenizer, end_effector_tokenizer=end_effector_tokenizer
    )


@fixture(scope="session")
def vima_policy() -> VIMAPolicy:
    return VIMAPolicy(embed_dim=512, xf_n_layers=3, sattn_n_heads=8, xattn_n_heads=8)
