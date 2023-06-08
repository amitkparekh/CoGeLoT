from pytest_cases import fixture

from cogelot.data.structures import View
from cogelot.modules.tokenizers import ImageTokenizer, MultimodalPromptTokenizer, TextTokenizer
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
        views=[View.FRONT, View.TOP],
    )


@fixture(scope="session")
def vima_policy() -> VIMAPolicy:
    return VIMAPolicy(embed_dim=512, xf_n_layers=3, sattn_n_heads=8, xattn_n_heads=8)