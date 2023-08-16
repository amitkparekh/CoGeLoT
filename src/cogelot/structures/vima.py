from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Literal, Self

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

N_DISCRETE_X_BINS: int = 50
N_DISCRETE_Y_BINS: int = 100
N_DISCRETE_Z_BINS: int = 50
N_DISCRETE_ROT_BINS: int = 50


class Partition(Enum):
    """Different levels of difficulty for the tasks."""

    placement_generalization = 1
    combinatorial_generalization = 2
    novel_object_generalization = 3
    novel_task_generalization = 4

    @classmethod
    def from_index(cls, index: int) -> Self:
        """Get the partition from the index."""
        return cls(index)


class Task(Enum):
    """Tasks in the VIMA dataset."""

    visual_manipulation = 1
    scene_understanding = 2
    rotate = 3
    rearrange = 4
    rearrange_then_restore = 5
    novel_adj = 6
    novel_noun = 7
    novel_adj_and_noun = 8
    twist = 9
    follow_motion = 10
    follow_order = 11
    sweep_without_exceeding = 12
    sweep_without_touching = 13
    same_texture = 14
    same_shape = 15
    manipulate_old_neighbor = 16
    pick_in_order_then_restore = 17

    # Old names that existed in the dataset and can cause issues, but are just the same as others.
    # Putting them below the others means that trying to refer to them will automatically redirect
    # to the correct one.
    same_color = 14  # noqa: PIE796
    same_profile = 15  # noqa: PIE796
    sweep = 12  # noqa: PIE796
    simple_manipulation = 1  # noqa: PIE796

    @classmethod
    def as_sorted_task_list(cls) -> list[str]:
        """Get the sorted task list."""
        return sorted(cls.__members__.keys())

    @classmethod
    def from_sorted_task_list_index(cls, index: int) -> Self:
        """Create by indexing from the sorted task list."""
        task_name = cls.as_sorted_task_list()[index]
        return cls[task_name]


class TaskGroup(Enum):
    """Grouping of tasks."""

    instruction_following = 1
    constraint_satisfaction = 2
    novel_concept_grounding = 3
    one_shot_imitation = 4
    rearrangement = 5
    require_memory = 6
    require_reasoning = 7


TaskPerGroup: Mapping[TaskGroup, list[Task]] = {
    TaskGroup.instruction_following: [
        Task.visual_manipulation,
        Task.scene_understanding,
        Task.rotate,
    ],
    TaskGroup.constraint_satisfaction: [Task.sweep_without_exceeding, Task.sweep_without_touching],
    TaskGroup.novel_concept_grounding: [
        Task.novel_adj_and_noun,
        Task.novel_adj,
        Task.novel_noun,
        Task.twist,
    ],
    TaskGroup.one_shot_imitation: [Task.follow_motion, Task.follow_order],
    TaskGroup.rearrangement: [Task.rearrange],
    TaskGroup.require_memory: [
        Task.manipulate_old_neighbor,
        Task.pick_in_order_then_restore,
        Task.rearrange_then_restore,
    ],
    TaskGroup.require_reasoning: [Task.same_texture, Task.same_shape],
}


EndEffector = Literal["suction", "spatula"]
PoseActionType = Literal["pose0_position", "pose0_rotation", "pose1_position", "pose1_rotation"]

PositionAxes = Literal["x", "y"]  # 'z' is not modelled and therefore not included here
RotationAxes = Literal["x", "y", "z", "w"]

AxesPerPoseActionType: dict[PoseActionType, type[PositionAxes | RotationAxes]] = {
    "pose0_position": PositionAxes,
    "pose0_rotation": RotationAxes,
    "pose1_position": PositionAxes,
    "pose1_rotation": RotationAxes,
}


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
