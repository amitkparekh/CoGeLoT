from typing import Any, ClassVar, Literal, cast, overload

import numpy as np
import torch
from torch.utils.data import default_collate

from cogelot.modules.tokenizers import EndEffectorTokenizer, TextTokenizer
from cogelot.modules.tokenizers.pose_action import PoseActionTokenizer
from cogelot.structures.common import ImageType, Observation, PromptAssets, View
from cogelot.structures.model import PreprocessedInstance, RawPromptTokenType
from cogelot.structures.vima import PoseAction, VIMAInstance
from vima.prepare_obs import ObsDict, prepare_obs
from vima.prepare_prompt import prepare_prompt
from vima.utils import DataDict, any_to_datadict


def convert_observations_to_their_format(
    *, observations: list[Observation], tokenized_end_effector: np.ndarray
) -> ObsDict:
    """Convert our observations to the format that VIMA uses."""
    observation_dicts = [observation.to_image_per_view_per_type() for observation in observations]
    rgb_dict: dict[Literal["top", "front"], np.ndarray] = {
        view.value: np.stack(
            [observation_dict[ImageType.rgb][view] for observation_dict in observation_dicts],
            axis=0,
        )
        for view in (View.front, View.top)
    }
    segm_dict: dict[Literal["top", "front"], np.ndarray] = {
        view.value: np.stack(
            [
                observation_dict[ImageType.segmentation][view]
                for observation_dict in observation_dicts
            ],
            axis=0,
        )
        for view in (View.front, View.top)
    }

    return ObsDict(ee=tokenized_end_effector, rgb=rgb_dict, segm=segm_dict)


class InstancePreprocessor:
    """Preprocess VIMA instances for the model.

    That means tokenizing them and getting them ready to just be embedded.
    """

    views: ClassVar[set[str]] = {"front", "top"}

    def __init__(
        self,
        *,
        text_tokenizer: TextTokenizer,
        end_effector_tokenizer: EndEffectorTokenizer,
        pose_action_tokenizer: PoseActionTokenizer,
    ) -> None:
        self.text_tokenizer = text_tokenizer
        self.end_effector_tokenizer = end_effector_tokenizer
        self.pose_action_tokenizer = pose_action_tokenizer

    def preprocess(self, instance: VIMAInstance) -> PreprocessedInstance:
        """Preprocess a single instance of the dataset."""
        # Prepare the prompt
        prompt = self.prepare_prompt(
            prompt=instance.prompt,
            prompt_assets=instance.prompt_assets.dict()["__root__"],
            object_ids_from_prompt_assets=instance.prompt_assets.all_object_ids,
        )

        # Prepare the observations
        observations = self.prepare_observations(
            observations=instance.observations,
            object_ids=instance.object_ids,
            end_effector=instance.end_effector_type,
        )

        actions = self.prepare_actions(instance.pose_actions)

        return PreprocessedInstance(
            task=instance.task,
            raw_prompts_token_type=prompt[0],
            word_batch=prompt[1],
            image_batch=prompt[2],
            observations=observations,
            actions=actions,
        )

    @overload
    def prepare_prompt(
        self,
        *,
        prompt: str,
        prompt_assets: dict[str, dict[str, Any]],
        object_ids_from_prompt_assets: set[int],
    ) -> tuple[RawPromptTokenType, torch.Tensor, DataDict]:
        ...  # noqa: WPS428

    @overload
    def prepare_prompt(
        self,
        *,
        prompt: str,
        prompt_assets: PromptAssets,
        object_ids_from_prompt_assets: None,
    ) -> tuple[RawPromptTokenType, torch.Tensor, DataDict]:
        ...  # noqa: WPS428

    def prepare_prompt(
        self,
        *,
        prompt: str,
        prompt_assets: dict[str, dict[str, Any]] | PromptAssets,
        object_ids_from_prompt_assets: set[int] | None,
    ) -> tuple[RawPromptTokenType, torch.Tensor, DataDict]:
        """Prepare the prompt for the model.

        Just take what VIMA does. This does not do any encoding, so it doesn't update any of the
        gradients or anything.
        """
        if object_ids_from_prompt_assets is None and isinstance(prompt_assets, PromptAssets):
            object_ids_from_prompt_assets = prompt_assets.all_object_ids

        if isinstance(prompt_assets, PromptAssets):
            prompt_assets = prompt_assets.dict()["__root__"]

        assert isinstance(prompt_assets, dict)
        assert isinstance(object_ids_from_prompt_assets, set)

        prepared_prompt = prepare_prompt(
            prompt=prompt,
            prompt_assets=prompt_assets,
            views=list(self.views),
            tokenizer=self.text_tokenizer.tokenizer,
            placeholders=list(self.text_tokenizer.all_placeholders),
            all_object_ids=object_ids_from_prompt_assets,
        )
        return cast(tuple[RawPromptTokenType, torch.Tensor, DataDict], prepared_prompt)

    def prepare_observations(
        self, observations: list[Observation], object_ids: set[int], end_effector: str
    ) -> DataDict:
        """Prepare observations for the model.

        Note: Does not need GPU.
        """
        tokenized_end_effector = self.tokenize_end_effector(end_effector, len(observations))
        their_observations = convert_observations_to_their_format(
            observations=observations, tokenized_end_effector=tokenized_end_effector
        )
        prepared_observations = prepare_obs(
            obs=their_observations,
            object_ids=list(object_ids),
        )
        return prepared_observations

    def tokenize_end_effector(self, end_effector: str, num_observations: int) -> np.ndarray:
        """Tokenize the end effector for all the observations."""
        tokenized_end_effector = self.end_effector_tokenizer.encode(end_effector)
        return np.array(tokenized_end_effector).repeat(num_observations)

    def prepare_actions(self, pose_actions: list[PoseAction]) -> DataDict:
        """Prepare the actions for the model."""
        tokenized_actions = self.pose_action_tokenizer.tokenize(pose_actions)
        action_dicts = [action.to_target_pose_action() for action in tokenized_actions]
        collated_actions = default_collate(action_dicts)
        return cast(DataDict, any_to_datadict(collated_actions).to_torch_tensor())
