import math
import random
from collections.abc import Callable

from cogelot.data.transforms.base import VIMAInstanceTransform
from cogelot.data.transforms.templates.formatter import TemplateFormatter
from cogelot.data.transforms.templates.replacer import extract_keys_from_original
from cogelot.modules.tokenizers.text import PLACEHOLDER_TOKENS
from cogelot.structures.common import PromptAssets
from cogelot.structures.vima import Task, VIMAInstance


def _extract_placeholders_from_instruction(instruction: str) -> set[str]:
    """Extract the placeholders from the instruction."""
    return {
        word[1:-1]
        for word in (i.strip().rstrip(".").rstrip(":") for i in instruction.lower().split(" "))
        if word.startswith("{") and word.endswith("}")
    }


PLACEHOLDER_NAME_TO_TOKEN = {token.asset_name: token for token in PLACEHOLDER_TOKENS}

PLACEHOLDER_ALTERNATIVES = {
    "base_obj": ["bounds"],
    "dragged_obj": ["swept_obj"],
}


POSSIBLE_DEGREES = [int(round(math.degrees(1 / 6 * math.pi * i), 0)) for i in range(1, 6)]


def _convert_1_to_3() -> str:
    """Use instruction from T3 in T1's environment."""
    return "Rotate the {dragged_obj_1} {angle_in_degree} degrees.".replace(
        "{angle_in_degree}",
        str(random.choice(POSSIBLE_DEGREES)),  # noqa: S311
    )


def _convert_12_to_13() -> str:
    """Use instruction from T13 in T12's environment."""
    return "Sweep {det} {swept_obj} into {bounds} without touching {constraint}."


def _convert_13_to_12() -> str:
    """Use instruction from T13 in T12's environment."""
    return "Sweep {det} {swept_obj} into {bounds} without exceeding {constraint}."


def _convert_14_to_15() -> str:
    """Use instruction from T14 in T15's environment."""
    return "Put all objects with the same profile as {base_obj} into it."


def _convert_15_to_14() -> str:
    """Use instruction from T15 in T14's environment."""
    return "Put all objects with the same texture as {base_obj} into it."


TASK_MAPPING: dict[Task, list[Callable[[], str]]] = {
    Task.visual_manipulation: [_convert_1_to_3],
    Task.sweep_without_exceeding: [_convert_12_to_13],
    Task.sweep_without_touching: [_convert_13_to_12],
    Task.same_texture: [_convert_14_to_15],
    Task.same_shape: [_convert_15_to_14],
}


class DifferentInstructionTransform(VIMAInstanceTransform):
    """Replace the task instruction with one from another task."""

    def __call__(self, instance: VIMAInstance) -> VIMAInstance:
        """Return the instance with a completely different language instruction for the visuals."""
        if instance.task not in TASK_MAPPING:
            raise NotImplementedError("This task is not supported.")

        new_instruction_template = random.choice(TASK_MAPPING[instance.task])()  # noqa: S311

        # Make the new instruction
        keys_from_original = extract_keys_from_original(
            instance.prompt, new_instruction_template, strict=False
        )
        new_instruction = TemplateFormatter().format(
            new_instruction_template, **keys_from_original
        )

        # Update the prompt assets
        placeholders = _extract_placeholders_from_instruction(new_instruction_template)
        prompt_assets = [
            asset for asset in instance.prompt_assets.root if asset.name in placeholders
        ]

        return instance.model_copy(
            deep=True,
            update={"prompt": new_instruction, "prompt_assets": PromptAssets(root=prompt_assets)},
        )
