from cogelot.data.transforms import (
    GobbledyGookPromptTokenTransform,
    GobbledyGookPromptWordTransform,
    NoopTransform,
)
from cogelot.modules.tokenizers.text import TextTokenizer
from cogelot.structures.vima import VIMAInstance


def test_noop_transform(vima_instance: VIMAInstance) -> None:
    transform = NoopTransform()
    new_instance = transform(vima_instance)

    assert new_instance == vima_instance


def test_gobbledygook_word_transform(vima_instance: VIMAInstance) -> None:
    gobbledygook_transform = GobbledyGookPromptWordTransform()
    new_instance = gobbledygook_transform(vima_instance)

    assert new_instance.prompt != vima_instance.prompt
    assert len(new_instance.prompt.split(" ")) == len(vima_instance.prompt.split(" "))


def test_gobbledygook_token_transform(
    vima_instance: VIMAInstance, text_tokenizer: TextTokenizer
) -> None:
    gobbledygook_transform = GobbledyGookPromptTokenTransform(text_tokenizer)
    new_instance = gobbledygook_transform(vima_instance)

    # Prompt must be different text
    assert new_instance.prompt != vima_instance.prompt
    # Prompts must have same number of tokens after tokenization
    assert len(text_tokenizer.tokenizer.encode(new_instance.prompt)) == len(
        text_tokenizer.tokenizer.encode(vima_instance.prompt)
    )
