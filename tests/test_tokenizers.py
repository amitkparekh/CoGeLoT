import torch

from cogelot.modules.tokenizers import ImageTokenizer, PoseActionTokenizer, TextTokenizer
from cogelot.structures.vima import VIMAInstance


def test_text_tokenizer_does_not_fail(
    vima_instance: VIMAInstance, text_tokenizer: TextTokenizer
) -> None:
    tokenized_output = text_tokenizer.tokenizer(vima_instance.prompt)
    assert tokenized_output is not None


def test_image_tokenizer_does_not_fail(
    vima_instance: VIMAInstance, image_tokenizer: ImageTokenizer
) -> None:
    tokens = [
        image_tokenizer.tokenize_observation(
            observation=observation, all_object_ids=vima_instance.object_ids
        )
        for observation in vima_instance.observations
    ]
    assert tokens is not None


def test_pose_action_tokenizer_works_properly(
    vima_instance: VIMAInstance, pose_action_tokenizer: PoseActionTokenizer
) -> None:
    original_pose_actions = vima_instance.pose_actions

    for original_pose_action in original_pose_actions:
        tokenized_action = next(
            iter(pose_action_tokenizer.tokenize([original_pose_action]))
        ).to_target_pose_action()

        continuous_actions = pose_action_tokenizer.convert_discrete_to_continuous(tokenized_action)
        discrete_actions = pose_action_tokenizer.convert_continuous_to_discrete(continuous_actions)

        for action_token, discrete_action in zip(
            tokenized_action.values(), discrete_actions.values()
        ):
            torch.testing.assert_allclose(action_token, discrete_action)

        # for original_action, continous_action in zip(
        #     original_pose_action.to_tensor().values(), continuous_actions.values()
        # ):
        #     # Continuous actions are not exactly the same as the original actions, but we want thme
        #     # to be close
        #     torch.testing.assert_allclose(original_action, continous_action, atol=0.02, rtol=1e-2)
