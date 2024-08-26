import pytest
from pytest_cases import parametrize

from cogelot.data.transforms import (
    GobbledyGookPromptTokenTransform,
    GobbledyGookPromptWordTransform,
    NoopTransform,
    RewordPromptTransform,
    TextualDescriptionTransform,
)

# from cogelot.data.transforms.instruction import InstructionReplacerTransform
from cogelot.data.transforms.textual import DescriptionModificationMethod
from cogelot.modules.tokenizers.text import TextTokenizer
from cogelot.structures.vima import VIMAInstance


def test_noop_transform_works(vima_instance: VIMAInstance) -> None:
    transform = NoopTransform()
    new_instance = transform(vima_instance)

    assert new_instance == vima_instance


def test_gobbledygook_word_transform_works(vima_instance: VIMAInstance) -> None:
    gobbledygook_transform = GobbledyGookPromptWordTransform()
    new_instance = gobbledygook_transform(vima_instance)

    assert new_instance.prompt != vima_instance.prompt
    assert len(new_instance.prompt.split(" ")) == len(vima_instance.prompt.split(" "))


def test_gobbledygook_token_transform_works(vima_instance: VIMAInstance) -> None:
    text_tokenizer = TextTokenizer("t5-base")
    gobbledygook_transform = GobbledyGookPromptTokenTransform(text_tokenizer)
    new_instance = gobbledygook_transform(vima_instance)

    # Prompt must be different text
    assert new_instance.prompt != vima_instance.prompt
    # Prompts must have same number of tokens after tokenization
    assert len(text_tokenizer.tokenizer.encode(new_instance.prompt)) == len(
        text_tokenizer.tokenizer.encode(vima_instance.prompt)
    )


def test_reword_transform_works(vima_instance: VIMAInstance) -> None:
    reword_transform = RewordPromptTransform(allow_original_reuse=False)
    new_instance = reword_transform(vima_instance)
    assert new_instance.prompt != vima_instance.prompt


@pytest.mark.skip
def test_count_all_reword_transforms(vima_instance: VIMAInstance) -> None:
    reword_transform = RewordPromptTransform()
    all_possible_templates = reword_transform.generate_all_possible_prompts(vima_instance)
    assert all_possible_templates


@parametrize(
    "modification_method",
    ["underspecify_nouns", "remove_textures", "remove_nouns", None],
)
def test_textual_description_transform_works(
    vima_instance: VIMAInstance,
    modification_method: DescriptionModificationMethod | None,
) -> None:
    if vima_instance.task in TextualDescriptionTransform.tasks_to_avoid:
        pytest.skip("This test is only for tasks that are not avoided by the transform")

    transform = TextualDescriptionTransform(
        description_modification_method=modification_method,
    )
    new_instance = transform(vima_instance)
    assert new_instance.prompt != vima_instance.prompt


def test_textual_description_transform_noops_correctly(
    vima_instance: VIMAInstance,
) -> None:
    if vima_instance.task not in TextualDescriptionTransform.tasks_to_avoid:
        pytest.skip("This test is only for tasks that are avoided by the transform")

    transform = TextualDescriptionTransform()
    new_instance = transform(vima_instance)
    assert new_instance == vima_instance


# def test_instruction_replacer_works(vima_instance: VIMAInstance) -> None:
#     vima_instance.partition = Partition.placement_generalization
#     transform = InstructionReplacerTransform()
#     new_instance = transform(vima_instance)

#     # Make sure the prompts are different
#     assert new_instance.prompt != vima_instance.prompt

#     # We need to make sure that the prompt assets all fit correctly otherwise the thing will crash
