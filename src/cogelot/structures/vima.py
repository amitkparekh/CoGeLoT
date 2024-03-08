from collections import Counter
from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Any, Literal, Self, TypeVar

import datasets
import polars as pl
import torch
from polars.type_aliases import SchemaDefinition
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
    ObjectDescription,
    Observation,
    PromptAsset,
    PromptAssets,
    PydanticHFDatasetMixin,
    PydanticTensor,
    Timestep,
)

Difficulty = Literal["easy", "medium", "hard"]

MODALITIES: tuple[Literal["segm", "rgb"], ...] = ("segm", "rgb")
VIDEO_FPS = 60
OUTPUT_VIDEO_NAME = "gui_record.mp4"
VIDEO_HEIGHT = 480

# Action decoder bins
N_DISCRETE_X_BINS: int = 50
N_DISCRETE_Y_BINS: int = 100
N_DISCRETE_Z_BINS: int = 50
N_DISCRETE_ROT_BINS: int = 50

# Action space boundaries
X_MIN = 0.25
X_MAX = 0.75
Y_MIN = -0.5
Y_MAX = 0.5
Z_MIN = 0
Z_MAX = 0.32
ROT_MIN = -1
ROT_MAX = 1

STARTING_POSITION_ENV = (0.5, 0)
STARTING_POSITION = (0.5, 0, Z_MAX)
STARTING_ROTATION = (0, 0, 0, 1)


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

    @classmethod
    def from_builtin_type(cls, value: str | int | Self) -> Self:  # noqa: WPS110
        """Get the task from the built-in type."""
        if isinstance(value, str):
            return cls[value]
        if isinstance(value, int):
            return cls(value)
        return value


class Task(Enum):
    """Tasks in the VIMA dataset."""

    visual_manipulation = 0
    scene_understanding = 1
    rotate = 2
    rearrange = 3
    rearrange_then_restore = 4
    novel_adj = 5
    novel_noun = 6
    novel_adj_and_noun = 7
    twist = 8
    follow_motion = 9
    follow_order = 10
    sweep_without_exceeding = 11
    sweep_without_touching = 12
    same_texture = 13
    same_shape = 14
    manipulate_old_neighbor = 15
    pick_in_order_then_restore = 16

    # Old names that existed in the dataset and can cause issues, but are just the same as others.
    # Putting them below the others means that trying to refer to them will automatically redirect
    # to the correct one.
    same_color = 13  # noqa: PIE796
    same_profile = 14  # noqa: PIE796
    sweep = 11  # noqa: PIE796
    simple_manipulation = 0  # noqa: PIE796

    @classmethod
    def dataset_feature(cls) -> datasets.ClassLabel:
        """Export the feature for the HF dataset."""
        return datasets.ClassLabel(names=cls._member_names_)

    @classmethod
    def minimum(cls) -> int:
        """Smallest value."""
        return min(enum.value for enum in cls)

    @classmethod
    def maximum(cls) -> int:
        """Largest value."""
        return max(enum.value for enum in cls)

    @classmethod
    def names_for_value(cls, value: int) -> list[str]:  # noqa: WPS110
        """Get all the names for the given value.

        This is useful so that if a given value has more than one name, we can get the names.
        """
        return [name for name, enum in Task.__members__.items() if enum.value == value]

    @classmethod
    def from_builtin_type(cls, value: str | int | Self) -> Self:  # noqa: WPS110
        """Get the task from the built-in type."""
        if isinstance(value, str):
            return cls[value]
        if isinstance(value, int):
            return cls(value)
        return value


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


def get_task_group_from_task(task: Task) -> TaskGroup:
    """Get the task group from the task."""
    for task_group, task_list in TaskPerGroup.items():
        if task in task_list:
            return task_group

    raise AssertionError("Task not found in list.")


EndEffector = Literal["suction", "spatula"]
PoseActionType = Literal["pose0_position", "pose0_rotation", "pose1_position", "pose1_rotation"]

PositionAxes = Literal["x", "y", "z"]
RotationAxes = Literal["x", "y", "z", "w"]

AxesPerPoseActionType: dict[PoseActionType, type[PositionAxes | RotationAxes]] = {
    "pose0_position": PositionAxes,  # type: ignore[dict-item]
    "pose0_rotation": RotationAxes,  # type: ignore[dict-item]
    "pose1_position": PositionAxes,  # type: ignore[dict-item]
    "pose1_rotation": RotationAxes,  # type: ignore[dict-item]
}

SEVector = tuple[float, float, float, float, float, float, float]


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

    @property
    def as_description(self) -> ObjectDescription:
        """Get the natural language name of the object."""
        return ObjectDescription(name=self.obj_name, texture=self.texture_name)


class PoseAction(Action, PydanticHFDatasetMixin):
    """Actions which are taken by the agent in the environment."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pose0_position: PydanticTensor
    pose1_position: PydanticTensor
    pose0_rotation: PydanticTensor
    pose1_rotation: PydanticTensor

    def __len__(self) -> int:
        """Return the number of actions."""
        return self.pose0_position.shape[0]

    @field_validator("pose0_position", "pose1_position")
    @classmethod
    def check_shape_of_pose_position(cls, tensor: torch.Tensor) -> torch.Tensor:
        """Verify the shape of the pose position."""
        assert tensor.shape == (3,), f"Expected shape (3,), got {tensor.shape}"
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
                "pose0_position": datasets.Sequence(datasets.Value("float32"), length=3),
                "pose0_rotation": datasets.Sequence(datasets.Value("float32"), length=4),
                "pose1_position": datasets.Sequence(datasets.Value("float32"), length=3),
                "pose1_rotation": datasets.Sequence(datasets.Value("float32"), length=4),
            }
        )

    @classmethod
    def get_null_action(cls, index: int = 0) -> Self:
        """Create an action that has no movement."""
        return cls(
            pose0_position=torch.tensor(STARTING_POSITION, dtype=torch.float32),
            pose0_rotation=torch.tensor(STARTING_ROTATION, dtype=torch.float32),
            pose1_position=torch.tensor(STARTING_POSITION, dtype=torch.float32),
            pose1_rotation=torch.tensor(STARTING_ROTATION, dtype=torch.float32),
            index=index,
        )

    @property
    def start_pose(self) -> SEVector:
        """Starting position as SE(3)."""
        return (*self.pose0_position.tolist(), *self.pose0_rotation.tolist())

    @property
    def end_pose(self) -> SEVector:
        """Ending position as SE(3)."""
        return (*self.pose1_position.tolist(), *self.pose1_rotation.tolist())

    def as_metadata(self) -> dict[Literal["start", "end"], SEVector]:
        """Get the metadata for the action."""
        return {"start": self.start_pose, "end": self.end_pose}

    def __repr__(self) -> str:
        """Representation of the action."""
        rounded_start_pose = tuple(round(x, 2) for x in self.start_pose)
        rounded_end_pose = tuple(round(x, 2) for x in self.end_pose)
        return f"PoseAction(idx={self.index}); {rounded_start_pose} -> {rounded_end_pose}"


T = TypeVar("T")


def maybe_convert_dict_list_to_list_dict(
    maybe_dict_list: dict[str, list[T]] | list[dict[str, T]],
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


def maybe_convert_batched_list_to_item(maybe_batched_list: T) -> T:
    """Convert a list of length 1 back into just the item."""
    if isinstance(maybe_batched_list, list) and len(maybe_batched_list) == 1:
        return maybe_batched_list[0]
    return maybe_batched_list


class VIMAInstanceMetadata(BaseModel):
    """Metadata for a VIMA instance."""

    index: int
    task: Task
    num_actions: int
    num_observations: int
    total_steps: int
    generation_seed: int
    end_effector_type: EndEffector
    difficulty: Difficulty
    prompt: str
    prompt_assets: dict[str, list[ObjectDescription]]
    scene_assets: list[ObjectDescription]
    actions: list[dict[Literal["start", "end"], SEVector]]

    # These are pre-computed
    num_objects: int
    shapes_in_scene: list[str]
    textures_in_scene: list[str]
    object_types_in_scene: list[str]

    @classmethod
    def polars_schema_override(cls) -> SchemaDefinition:
        """Return override for polars schema."""
        return {
            "index": pl.Int64,
            "task": pl.Int8,
            "num_objects": pl.Int64,
            "num_actions": pl.Int64,
            "num_observations": pl.Int32,
            "total_steps": pl.Int32,
            "generation_seed": pl.Int64,
            "end_effector_type": pl.String,
            "difficulty": pl.String,
            "prompt": pl.String,
            "scene_assets": pl.List(pl.Struct({"name": pl.String, "texture": pl.String})),
            "actions": pl.List(
                pl.Struct(
                    {"start": pl.Array(pl.Float64, width=7), "end": pl.Array(pl.Float64, width=7)}
                )
            ),
        }


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
        BeforeValidator(maybe_convert_batched_list_to_item),
        BeforeValidator(lambda task: int(task.item()) if isinstance(task, torch.Tensor) else task),
        BeforeValidator(lambda task: Task(task) if isinstance(task, int) else task),
        PlainSerializer(lambda task: task.value, return_type=int),
    ]

    # If doesn't exist, default to easy since that appears to be the default stated
    difficulty: Annotated[Difficulty, BeforeValidator(maybe_convert_batched_list_to_item)] = "easy"

    # If the incoming data is a tensor, make sure to convert it to an integer
    index: Annotated[
        int, BeforeValidator(maybe_convert_batched_list_to_item), BeforeValidator(int)
    ]
    total_steps: Annotated[
        int, BeforeValidator(maybe_convert_batched_list_to_item), BeforeValidator(int)
    ]

    end_effector_type: Annotated[EndEffector, BeforeValidator(maybe_convert_batched_list_to_item)]

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

    prompt: Annotated[str, BeforeValidator(maybe_convert_batched_list_to_item)]
    prompt_assets: Annotated[PromptAssets, BeforeValidator(maybe_convert_dict_list_to_list_dict)]

    # Seed used when generating the instance
    generation_seed: Annotated[int, BeforeValidator(maybe_convert_batched_list_to_item)]

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
    def num_shapes_in_scene(self) -> Counter[str]:
        """How many of each shape is in the scene?"""
        return Counter(self._shapes_in_scene)

    @property
    def num_textures_in_scene(self) -> Counter[str]:
        """How many of each texture is in the scene?"""
        return Counter(self._textures_in_scene)

    @property
    def num_object_types_in_scene(self) -> Counter[str]:
        """How many of each object type is in the scene?"""
        return Counter(self._object_types_in_scene)

    @classmethod
    def dataset_features(cls) -> datasets.Features:
        """Get the dataset features for a VIMA instance."""
        return datasets.Features(
            {
                "index": datasets.Value("int64"),
                "task": Task.dataset_feature(),
                "difficulty": datasets.Value("string"),
                "object_metadata": datasets.Sequence(ObjectMetadata.dataset_features()),
                "total_steps": datasets.Value("int64"),
                "end_effector_type": datasets.Value("string"),
                "observations": datasets.Sequence(Observation.dataset_features()),
                "pose_actions": datasets.Sequence(PoseAction.dataset_features()),
                "prompt": datasets.Value("string"),
                "prompt_assets": datasets.Sequence(PromptAsset.dataset_features()),
                "generation_seed": datasets.Value("int64"),
            }
        )

    def to_metadata(self) -> VIMAInstanceMetadata:
        """Pre-compute the metadata for easier statistics gathering."""
        return VIMAInstanceMetadata(
            index=self.index,
            task=self.task,
            num_objects=self.num_objects,
            num_actions=self.num_actions,
            difficulty=self.difficulty,
            num_observations=self.num_observations,
            total_steps=self.total_steps,
            generation_seed=self.generation_seed,
            end_effector_type=self.end_effector_type,
            prompt=self.prompt,
            prompt_assets={asset.name: asset.descriptions for asset in self.prompt_assets.root},
            scene_assets=[obj.as_description for obj in self.object_metadata],
            actions=[action.as_metadata() for action in self.pose_actions],
            shapes_in_scene=self._shapes_in_scene,
            textures_in_scene=self._textures_in_scene,
            object_types_in_scene=self._object_types_in_scene,
        )

    @property
    def _shapes_in_scene(self) -> list[str]:
        """Shapes in the scene."""
        return [obj.as_description.name for obj in self.object_metadata]

    @property
    def _textures_in_scene(self) -> list[str]:
        """Textures in the scene."""
        return [obj.as_description.texture for obj in self.object_metadata]

    @property
    def _object_types_in_scene(self) -> list[str]:
        """Object types in the scene."""
        return [str(obj.as_description) for obj in self.object_metadata]
