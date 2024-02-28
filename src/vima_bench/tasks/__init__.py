import importlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .partition_files import *
from .task_suite import *
from .task_suite.base import BaseTask

__all__ = ["ALL_TASKS", "ALL_PARTITIONS", "get_partition_to_specs"]


@lru_cache(maxsize=1)
def _get_all_tasks() -> dict[str, list[BaseTask]]:
    return {
        "instruction_following": [
            SimpleManipulation,
            SceneUnderstanding,
            Rotate,
        ],
        "constraint_satisfaction": [
            WithoutExceeding,
            WithoutTouching,
        ],
        "novel_concept_grounding": [
            NovelAdjAndNoun,
            NovelAdj,
            NovelNoun,
            Twist,
        ],
        "one_shot_imitation": [
            FollowMotion,
            FollowOrder,
        ],
        "rearrangement": [Rearrange],
        "require_memory": [
            ManipulateOldNeighbor,
            PickInOrderThenRestore,
            RearrangeThenRestore,
        ],
        "require_reasoning": [
            SameColor,
            SameProfile,
        ],
    }


@lru_cache(maxsize=1)
def _get_all_tasks_flattened() -> dict[str, BaseTask]:
    return {
        f"{group}/{task.task_name}": task
        for group, tasks in _get_all_tasks().items()
        for task in tasks
    }


@lru_cache(maxsize=1)
def get_all_task_sub_names() -> list[str]:
    return [task.task_name for tasks in _get_all_tasks().values() for task in tasks]


@lru_cache(maxsize=1)
def _partition_file_path(fname: str) -> str:
    path = importlib.resources.files("vima_bench.tasks.partition_files")
    assert isinstance(path, Path)
    return path.joinpath(fname).resolve()


@lru_cache(maxsize=1)
def _load_partition_file(fname: str):
    file = _partition_file_path(fname)
    partition = OmegaConf.to_container(OmegaConf.load(file), resolve=True)
    partition_keys = set(partition.keys())
    for k in partition_keys:
        if k not in _ALL_TASK_SUB_NAMES:
            partition.pop(k)
    return partition


@lru_cache(maxsize=1)
def get_partition_to_specs() -> dict[str, dict[str, Any]]:
    return {
        "train": _load_partition_file("train.yaml"),
        "test": {
            "placement_generalization": _load_partition_file("placement_generalization.yaml"),
            "combinatorial_generalization": _load_partition_file(
                "combinatorial_generalization.yaml"
            ),
            "novel_object_generalization": _load_partition_file(
                "novel_object_generalization.yaml"
            ),
            "novel_task_generalization": _load_partition_file("novel_task_generalization.yaml"),
        },
    }


# # train
# TRAIN_PARTITION = _load_partition_file("train.yaml")

# # test
# PLACEMENT_GENERALIZATION = _load_partition_file("placement_generalization.yaml")
# COMBINATORIAL_GENERALIZATION = _load_partition_file("combinatorial_generalization.yaml")
# NOVEL_OBJECT_GENERALIZATION = _load_partition_file("novel_object_generalization.yaml")
# NOVEL_TASK_GENERALIZATION = _load_partition_file("novel_task_generalization.yaml")


ALL_PARTITIONS = [
    "placement_generalization",
    "combinatorial_generalization",
    "novel_object_generalization",
    "novel_task_generalization",
]


ALL_TASKS = _get_all_tasks_flattened()
_ALL_TASK_SUB_NAMES = get_all_task_sub_names()
