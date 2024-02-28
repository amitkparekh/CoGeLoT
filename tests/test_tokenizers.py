from typing import cast

import torch

from cogelot.modules.tokenizers import PoseActionTokenizer, TextTokenizer
from cogelot.structures.vima import PoseActionType, VIMAInstance


def test_text_tokenizer_does_not_fail(
    vima_instance: VIMAInstance, text_tokenizer: TextTokenizer
) -> None:
    tokenized_output = text_tokenizer.tokenizer(vima_instance.prompt)
    assert tokenized_output is not None


def test_pose_action_tokenizer_denormalizes_actions(
    vima_instance: VIMAInstance, pose_action_tokenizer: PoseActionTokenizer
) -> None:
    original_continuous_actions = [
        cast(dict[PoseActionType, torch.Tensor], action.model_dump(exclude={"index"}))
        for action in vima_instance.pose_actions
    ]

    for original_continuous_action in original_continuous_actions:
        # Make sure de-normalized actions are the same as the original actions
        denormalised_continuous_action = (
            pose_action_tokenizer._restore_rescaled_continuous_to_correct_range(
                pose_action_tokenizer.normalize_continuous_actions(original_continuous_action)
            )
        )
        torch.testing.assert_close(original_continuous_action, denormalised_continuous_action)


def test_pose_action_tokenizer_create_reversible_discrete_actions(
    vima_instance: VIMAInstance, pose_action_tokenizer: PoseActionTokenizer
) -> None:
    original_continuous_actions = [
        cast(dict[PoseActionType, torch.Tensor], action.model_dump(exclude={"index"}))
        for action in vima_instance.pose_actions
    ]

    for original_continuous_action in original_continuous_actions:
        # Make sure the discrete actions can be converted back reliably
        continuous_actions = pose_action_tokenizer.convert_discrete_to_continuous(
            pose_action_tokenizer.convert_continuous_to_discrete(original_continuous_action)
        )
        torch.testing.assert_close(
            continuous_actions, original_continuous_action, atol=0.02, rtol=0.02
        )
