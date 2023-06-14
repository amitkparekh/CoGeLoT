from cogelot.modules.tokenizers import ImageTokenizer, TextTokenizer
from cogelot.structures.vima import VIMAInstance


def test_text_tokenizer_does_not_fail(
    normalized_instance: VIMAInstance, text_tokenizer: TextTokenizer
) -> None:
    tokenized_output = text_tokenizer.tokenizer(normalized_instance.prompt)
    assert tokenized_output is not None


def test_image_tokenizer_does_not_fail(
    normalized_instance: VIMAInstance, image_tokenizer: ImageTokenizer
) -> None:
    tokens = [
        image_tokenizer.tokenize_observation(
            observation=observation, all_object_ids=normalized_instance.object_ids
        )
        for observation in normalized_instance.observations
    ]
    assert tokens is not None
