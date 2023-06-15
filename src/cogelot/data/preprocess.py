from __future__ import annotations

from typing import TYPE_CHECKING

from cogelot.structures.model import PreprocessedInstance


if TYPE_CHECKING:
    from cogelot.modules.tokenizers import (
        MultimodalHistoryTokenizer,
        MultimodalPromptTokenizer,
        PoseActionTokenizer,
    )
    from cogelot.structures.vima import VIMAInstance


class InstancePreprocessor:
    """Preprocess VIMA instances for the model.

    That means tokenizing them and getting them ready to just be embedded.
    """

    def __init__(
        self,
        *,
        multimodal_prompt_tokenizer: MultimodalPromptTokenizer,
        multimodal_history_tokenizer: MultimodalHistoryTokenizer,
        pose_action_tokenizer: PoseActionTokenizer,
    ) -> None:
        self.multimodal_prompt_tokenizer = multimodal_prompt_tokenizer
        self.multimodal_history_tokenizer = multimodal_history_tokenizer
        self.pose_action_tokenizer = pose_action_tokenizer

    def process(self, instance: VIMAInstance) -> PreprocessedInstance:
        """Preprocess a single instance of the dataset."""
        tokenized_prompt = self.multimodal_prompt_tokenizer.tokenize(
            prompt=instance.prompt, assets=instance.prompt_assets
        )
        tokenized_history = self.multimodal_history_tokenizer.tokenize(
            observations=instance.observations,
            pose_actions=instance.actions_history,
            end_effector=instance.end_effector_type,
            all_object_ids=instance.object_ids,
        )
        tokenized_target = self.pose_action_tokenizer.tokenize([instance.target_action])[0]

        return PreprocessedInstance(
            prompt=tokenized_prompt,
            history=tokenized_history,
            target=tokenized_target,
        )
