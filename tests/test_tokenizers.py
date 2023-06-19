from cogelot.modules.tokenizers import ImageTokenizer, PoseActionTokenizer, TextTokenizer
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


def test_pose_action_tokenizer_works_properly(
    normalized_instance: VIMAInstance, pose_action_tokenizer: PoseActionTokenizer
) -> None:
    tokenized_actions = pose_action_tokenizer.tokenize(normalized_instance.pose_actions)

    assert tokenized_actions

    # Make sure each is discrete and not a floating point
    for token in tokenized_actions:
        for tensor in token.to_target_pose_action().values():
            assert not tensor.is_floating_point()
