from collections.abc import Iterable
from typing import Any

from vima_bench.tasks.components.placeholders import (
    Placeholder,
    PlaceholderObj,
    PlaceholderTexture,
)
from vima_bench.tasks.task_suite.base import BaseTask


def parse_placeholder_obj(placeholder: PlaceholderObj) -> dict[str, Any]:
    """Parse the placeholder obj."""
    return {
        "name": placeholder.name,
        "obj_id": placeholder.obj_id,
        "urdf": placeholder.urdf,
        "novel_name": placeholder.novel_name,
        "obj_position": placeholder.obj_position,
        "obj_orientation": placeholder.obj_orientation,
        "alias": placeholder.alias,
        "color": placeholder.color.name,
        "scaling": placeholder.global_scaling,
    }


def parse_placeholder_texture(placeholder: PlaceholderTexture) -> dict[str, Any]:
    """Parse the placeholder texture."""
    return {
        "name": placeholder.name,
        "color_value": placeholder.color_value,
        "alias": placeholder.alias,
        "novel_name": placeholder.novel_name,
    }


def parse_placeholders(
    placeholders: Iterable[Placeholder],
) -> list[dict[str, Any]]:
    """Parse the placeholders."""
    parsed_list = []
    for placeholder in placeholders:
        if isinstance(placeholder, PlaceholderObj):
            parsed_list.append(parse_placeholder_obj(placeholder))
        if isinstance(placeholder, PlaceholderTexture):
            parsed_list.append(parse_placeholder_texture(placeholder))
    return parsed_list


def parse_base_task(task: BaseTask) -> dict[str, Any]:
    """Parse the base task."""
    task_meta = task.task_meta

    # This one causes a bunch of issues with logging tables and I don't even know what it means, so
    # we just don't log it.
    task_meta.pop("sample_prob")

    return {
        "task_meta": task_meta,
        "placeholders": parse_placeholders(task.placeholders.values()),
    }
