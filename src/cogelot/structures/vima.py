from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Any, Literal, Self, TypeVar

import datasets
import torch
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    field_validator,
)

from cogelot.structures.common import (
    Action,
    Observation,
    PromptAsset,
    PromptAssets,
    PydanticHFDatasetMixin,
    Timestep,
)


SEED = 42
MODALITIES: tuple[Literal["segm", "rgb"], ...] = ("segm", "rgb")
VIDEO_FPS = 60
OUTPUT_VIDEO_NAME = "gui_record.mp4"
VIDEO_HEIGHT = 480


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

    @classmethod
    def dataset_feature(cls) -> datasets.ClassLabel:
        """Export the feature for the HF dataset."""
        return datasets.ClassLabel(names=cls.as_sorted_task_list())


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


class ObjectMetadata(BaseModel, PydanticHFDatasetMixin):
    """Metadata for a given object."""

    obj_id: int
    obj_name: str
    obj_asset_name: str
    texture_name: str

    @classmethod
    def dataset_features(cls) -> datasets.Features:
        """Export the features schema for the HF dataset."""
        return datasets.Features(
            {
                "obj_id": datasets.Value("int64"),
                "obj_name": datasets.Value("string"),
                "obj_asset_name": datasets.Value("string"),
                "texture_name": datasets.Value("string"),
            }
        )


class PoseAction(Action, PydanticHFDatasetMixin):
    """Actions which are taken by the agent in the environment."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pose0_position: torch.Tensor
    pose1_position: torch.Tensor
    pose0_rotation: torch.Tensor
    pose1_rotation: torch.Tensor

    def __len__(self) -> int:
        """Return the number of actions."""
        return self.pose0_position.shape[0]

    @field_validator("pose0_position", "pose1_position")
    @classmethod
    def check_shape_of_pose_position(cls, tensor: torch.Tensor) -> torch.Tensor:
        """Verify the shape of the pose position."""
        assert tensor.shape == (2,), f"Expected shape (2,), got {tensor.shape}"
        return tensor

    @field_validator("pose0_rotation", "pose1_rotation")
    @classmethod
    def check_shape_of_pose_rotation(cls, tensor: torch.Tensor) -> torch.Tensor:
        """Verify the shape of the pose rotation."""
        assert tensor.shape == (4,), f"Expected shape (4,), got {tensor.shape}"
        return tensor

    @classmethod
    def dataset_features(cls) -> datasets.Features:
        """Export the features schema for the HF dataset."""
        return datasets.Features(
            {
                **Action.dataset_features(),
                "pose0_position": datasets.Sequence(datasets.Value("float32"), length=2),
                "pose0_rotation": datasets.Sequence(datasets.Value("float32"), length=4),
                "pose1_position": datasets.Sequence(datasets.Value("float32"), length=2),
                "pose1_rotation": datasets.Sequence(datasets.Value("float32"), length=4),
            }
        )


T = TypeVar("T")


def maybe_convert_dict_list_to_list_dict(
    maybe_dict_list: dict[str, list[T]] | list[dict[str, T]]
) -> list[dict[str, Any]]:
    """Convert a list of dicts to a dict of lists.

    Taken from: https://stackoverflow.com/a/33046935.

    This function goes from a dict of lists, to a list of dicts.
    For example,
    Before: `{'a': [0, 1], 'b': [2, 3]}`
    After: `[{'a': 0, 'b': 2}, {'a': 1, 'b': 3}]`
    """
    if isinstance(maybe_dict_list, dict):
        maybe_dict_list = [
            dict(zip(maybe_dict_list, dict_values, strict=True))
            for dict_values in zip(*maybe_dict_list.values(), strict=True)
        ]
    return maybe_dict_list


class VIMAInstance(BaseModel, PydanticHFDatasetMixin):
    """A single instance of the VIMA dataset, merging all the files into a single object.

    Yes, I know that this looks incredibly complicated at a glance, but I'm hoping that it doesn't
    with some explanation. I have leaned heavily on Pydantic's Validators through typing's
    Annotated to do a lot of the heavy lifting. If you know Pydantic v2, you'll be fine.

    The purpose of all these additional validators is to make it easier to parse the data from a HF
    dataset.
    """

    task: Annotated[
        Task,
        BeforeValidator(lambda task: int(task.item()) if isinstance(task, torch.Tensor) else task),
        BeforeValidator(
            lambda task: Task.from_sorted_task_list_index(task) if isinstance(task, int) else task
        ),
        PlainSerializer(lambda task: task.value, return_type=int),
    ]

    # If the incoming data is a tensor, make sure to convert it to an integer
    index: Annotated[int, BeforeValidator(int)]
    total_steps: Annotated[int, BeforeValidator(int)]

    end_effector_type: EndEffector

    # If the incoming data is a dict, make sure to convert it to a list of dicts, essentially
    # unwrapping the thing.
    object_metadata: Annotated[
        list[ObjectMetadata],
        BeforeValidator(maybe_convert_dict_list_to_list_dict),
    ]
    observations: Annotated[
        list[Observation],
        BeforeValidator(maybe_convert_dict_list_to_list_dict),
    ] = Field(default_factory=list)
    pose_actions: Annotated[
        list[PoseAction],
        BeforeValidator(maybe_convert_dict_list_to_list_dict),
    ] = Field(default_factory=list)

    prompt: str
    prompt_assets: Annotated[PromptAssets, BeforeValidator(maybe_convert_dict_list_to_list_dict)]

    @field_validator("pose_actions", "observations")
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

    @classmethod
    def dataset_features(cls) -> datasets.Features:
        """Get the dataset features for a VIMA instance."""
        return datasets.Features(
            {
                "index": datasets.Value("int64"),
                "task": Task.dataset_feature(),
                "object_metadata": datasets.Sequence(ObjectMetadata.dataset_features()),
                "total_steps": datasets.Value("int64"),
                "end_effector_type": datasets.Value("string"),
                "observations": datasets.Sequence(Observation.dataset_features()),
                "pose_actions": datasets.Sequence(PoseAction.dataset_features()),
                "prompt": datasets.Value("string"),
                "prompt_assets": datasets.Sequence(PromptAsset.dataset_features()),
            }
        )
