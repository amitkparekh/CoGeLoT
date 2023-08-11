from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Self, get_args

import numpy as np
import orjson
import torch
from pydantic import BaseModel, Field, validator
from pydantic_numpy import NDArray

from cogelot.common.io import load_json, orjson_dumps, save_json
from cogelot.structures.common import Action, Assets, Observation, Timestep


SEED = 42
MODALITIES: tuple[Literal["segm", "rgb"], ...] = ("segm", "rgb")
VIDEO_FPS = 60
OUTPUT_VIDEO_NAME = "gui_record.mp4"
VIDEO_HEIGHT = 480
VIDEO_WIDTH = 640

Partition = Literal[
    "placement_generalization",
    "combinatorial_generalization",
    "novel_object_generalization",
    "novel_task_generalization",
]

PARTITION_PER_LEVEL: Mapping[Literal[1, 2, 3, 4], Partition] = MappingProxyType(
    {
        1: "placement_generalization",
        2: "combinatorial_generalization",
        3: "novel_object_generalization",
        4: "novel_task_generalization",
    }
)

LEVEL_PER_PARTITION: Mapping[Partition, Literal[1, 2, 3, 4]] = MappingProxyType(
    {partition: index for index, partition in PARTITION_PER_LEVEL.items()}
)

Task = Literal[
    "follow_motion",
    "follow_order",
    "manipulate_old_neighbor",
    "novel_adj_and_noun",
    "novel_adj",
    "novel_noun",
    "pick_in_order_then_restore",
    "rearrange_then_restore",
    "rearrange",
    "rotate",
    "same_color",
    "same_profile",
    "same_shape",
    "same_texture",
    "scene_understanding",
    "simple_manipulation",
    "sweep_without_exceeding",
    "sweep_without_touching",
    "sweep",
    "twist",
    "visual_manipulation",
]

TASK_PER_INDEX: Mapping[int, Task] = MappingProxyType(
    {
        1: "visual_manipulation",
        2: "scene_understanding",
        3: "rotate",
        4: "rearrange",
        5: "rearrange_then_restore",
        6: "novel_adj",
        7: "novel_noun",
        8: "novel_adj_and_noun",
        9: "twist",
        10: "follow_motion",
        11: "follow_order",
        12: "sweep_without_exceeding",
        13: "sweep_without_touching",
        14: "same_texture",
        15: "same_shape",
        16: "manipulate_old_neighbor",
        17: "pick_in_order_then_restore",
    }
)

INDEX_PER_TASK: Mapping[Task, int] = MappingProxyType(
    {task: index for index, task in TASK_PER_INDEX.items()}
)

TaskGroup = Literal[
    "instruction_following",
    "constraint_satisfaction",
    "novel_concept_grounding",
    "one_shot_imitation",
    "rearrangement",
    "require_memory",
    "require_reasoning",
]

TaskPerGroup: Mapping[TaskGroup, list[Task]] = {
    "instruction_following": ["visual_manipulation", "scene_understanding", "rotate"],
    "constraint_satisfaction": ["sweep_without_exceeding", "sweep_without_touching"],
    "novel_concept_grounding": ["novel_adj_and_noun", "novel_adj", "novel_noun", "twist"],
    "one_shot_imitation": ["follow_motion", "follow_order"],
    "rearrangement": ["rearrange"],
    "require_memory": [
        "manipulate_old_neighbor",
        "pick_in_order_then_restore",
        "rearrange_then_restore",
    ],
    "require_reasoning": ["same_texture", "same_shape"],
}

EndEffector = Literal["suction", "spatula"]
PoseActionType = Literal["pose0_position", "pose0_rotation", "pose1_position", "pose1_rotation"]

SortedTaskList: list[Task] = sorted(get_args(Task))


class ObjectMetadata(BaseModel):
    """Metadata for a given object."""

    obj_id: int
    obj_name: str
    obj_asset_name: str
    texture_name: str


class PoseAction(Action, arbitrary_types_allowed=True):
    """Actions which are taken by the agent in the environment."""

    pose0_position: NDArray
    pose1_position: NDArray
    pose0_rotation: NDArray
    pose1_rotation: NDArray

    def __len__(self) -> int:
        """Return the number of actions."""
        return self.pose0_position.shape[0]

    @property
    def is_continuous(self) -> bool:
        """Determine if the actions are currently continuous or not."""
        array_is_continuous_tracker: list[bool] = []

        # For every value in the class, make sure its a numpy array, and then check if its
        # a float or not.
        for possible_array in dict(self).values():
            if isinstance(possible_array, np.ndarray):
                is_float = np.issubdtype(possible_array.dtype, np.floating)
                cast_to_float_is_identical_to_original = bool(
                    np.all(possible_array.astype(float) == possible_array)
                )
                array_is_continuous_tracker.append(
                    cast_to_float_is_identical_to_original and is_float
                )

        # Make sure all are true
        if sum(array_is_continuous_tracker) > 0 and not all(array_is_continuous_tracker):
            raise ValueError("Some are continuous?")

        return all(array_is_continuous_tracker)

    def to_tensor(self) -> dict[PoseActionType, torch.Tensor]:
        """Convert the actions to a tensor dict."""
        return {
            "pose0_position": torch.from_numpy(self.pose0_position),
            "pose1_position": torch.from_numpy(self.pose1_position),
            "pose0_rotation": torch.from_numpy(self.pose0_rotation),
            "pose1_rotation": torch.from_numpy(self.pose1_rotation),
        }


class ActionBounds(BaseModel, arbitrary_types_allowed=True):
    """Bounds for the actions."""

    low: NDArray
    high: NDArray


class VIMAInstance(BaseModel):
    """A single instance of the VIMA dataset, merging all the files into a single object."""

    index: int
    task: Task

    total_steps: int

    object_metadata: list[ObjectMetadata]

    end_effector_type: EndEffector

    action_bounds: ActionBounds
    observations: list[Observation] = Field(default_factory=list)
    pose_actions: list[PoseAction] = Field(default_factory=list)

    prompt: str
    prompt_assets: Assets

    class Config:
        """Pydantic config."""

        json_loads = orjson.loads
        json_dumps = orjson_dumps
        arbitrary_types_allowed = True

    @validator("pose_actions", "observations", allow_reuse=True)
    @classmethod
    def sort_by_index(cls, indexed_steps: list[Timestep]) -> list[Timestep]:
        """Sort the steps by index."""
        indexed_steps.sort(key=lambda step: step.index)
        return indexed_steps

    @property
    def num_actions(self) -> int:
        """Number of actions in the instance."""
        return len(self.pose_actions)

    @property
    def num_observations(self) -> int:
        """Number of observations in the instance."""
        return len(self.observations)

    @property
    def num_objects(self) -> int:
        """Get the number of objects in the instance."""
        return len(self.object_metadata)

    @property
    def object_ids(self) -> set[int]:
        """Get the object ids."""
        return {obj.obj_id for obj in self.object_metadata}

    @property
    def file_name(self) -> str:
        """Get the file name."""
        return f"{self.task}/{self.index}.json"

    def save(self, output_dir: Path, *, compress: bool = False) -> Path:
        """Save the file to the output dir."""
        instance_path = output_dir.joinpath(self.file_name)
        instance_path.parent.mkdir(parents=True, exist_ok=True)
        output_path = save_json(self.dict(), instance_path, compress=compress)
        return output_path

    @classmethod
    def load(cls, instance_path: Path) -> Self:
        """Load the instance from the file."""
        return cls.parse_obj(load_json(instance_path))
