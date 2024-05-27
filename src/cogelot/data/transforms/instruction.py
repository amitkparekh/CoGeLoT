import math
import random
from typing import ClassVar

from loguru import logger

from cogelot.data.transforms.base import VIMAInstanceTransform
from cogelot.environment.vima import get_task_kwargs
from cogelot.modules.tokenizers.text import PLACEHOLDER_TOKENS
from cogelot.structures.common import PromptAssets
from cogelot.structures.vima import Partition, Task, VIMAInstance
from vima_bench.tasks import ALL_TASKS as _ALL_TASKS
from vima_bench.tasks.task_suite.base import BaseTask


def _get_all_tasks() -> dict[str, type[BaseTask]]:
    """Get all tasks."""
    all_tasks = _ALL_TASKS.copy()
    all_tasks.update({k.split("/")[1]: v for k, v in all_tasks.items()})
    return all_tasks


def _generate_other_instruction(task_to_avoid: Task, partition: Partition) -> str:
    """Generate an instruction for the task."""
    task = task_to_avoid
    task_kwargs = None
    while task == task_to_avoid:
        # Pick a new task
        proposed_task = random.choice(list(Task))  # noqa: S311

        try:
            task_kwargs = get_task_kwargs(partition, proposed_task)
        except KeyError:
            logger.error(f"Could not get task kwargs for {proposed_task.name} in {partition.name}")
        else:
            task = proposed_task

    created_task = _get_all_tasks()[task.name](debug=False, **(task_kwargs or {}))
    return created_task.prompt_template


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


def _map_placeholders_from_old_to_new(
    old_placeholders: set[str], new_placeholders: set[str]
) -> dict[str, str]:
    """Map the old placeholders to the new ones."""
    placeholder_mapping = {}

    intersecting_placeholders = old_placeholders.intersection(new_placeholders)
    for placeholder in intersecting_placeholders:
        placeholder_mapping[placeholder] = placeholder

    remaining_placeholders = old_placeholders - intersecting_placeholders

    raise NotImplementedError


ROTATE_INSTRUCTION = "Rotate the {dragged_obj_1} {angle_in_degree} degrees."
POSSIBLE_DEGREES = [1 / 6 * math.pi * i for i in range(1, 6)]


class InstructionReplacerTransform(VIMAInstanceTransform):
    """Replace the task instruction with one from any other task."""

    disabled_tasks: ClassVar[set[Task]] = {
        Task.novel_adj,
        Task.novel_noun,
        Task.novel_adj_and_noun,
        Task.twist,
    }

    def __call__(self, instance: VIMAInstance) -> VIMAInstance:
        """Return the instance with a completely different language instruction for the visuals."""
        if instance.task != Task.visual_manipulation:
            raise NotImplementedError("Other tasks are not supported rn.")

        # Make sure its mapped right
        new_instruction = ROTATE_INSTRUCTION.replace(
            "{angle_in_degree}",
            str(random.choice(POSSIBLE_DEGREES)),  # noqa: S311
        )

        placeholders = _extract_placeholders_from_instruction(new_instruction)

        prompt_assets = [
            asset for asset in instance.prompt_assets.root if asset.name in placeholders
        ]

        return instance.model_copy(
            deep=True,
            update={"prompt": new_instruction, "prompt_assets": PromptAssets(root=prompt_assets)},
        )
