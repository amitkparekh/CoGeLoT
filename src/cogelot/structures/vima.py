from __future__ import annotations

from typing import Literal

import orjson
import torch
from loguru import logger
from pydantic import BaseModel, validator
from pydantic_numpy import NDArray

from cogelot.common.io import orjson_dumps
from cogelot.structures.common import Action, Assets, Observation, Timestep


EndEffector = Literal["suction", "spatula"]
PoseActionType = Literal["pose0_position", "pose0_rotation", "pose1_position", "pose1_rotation"]


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

    def to_tensor(self) -> dict[PoseActionType, torch.Tensor]:
        """Convert the actions to a tensor dict."""
        return {
            "pose0_position": torch.from_numpy(self.pose0_position),
            "pose1_position": torch.from_numpy(self.pose1_position),
            "pose0_rotation": torch.from_numpy(self.pose0_rotation),
            "pose1_rotation": torch.from_numpy(self.pose1_rotation),
        }


class VIMAInstance(BaseModel):
    """A single instance of the VIMA dataset, merging all the files into a single object."""

    index: int
    task: str

    total_steps: int

    object_metadata: list[ObjectMetadata]

    end_effector_type: EndEffector

    pose_actions: list[PoseAction]
    observations: list[Observation]

    prompt: str
    prompt_assets: Assets

    class Config:
        """Pydantic config."""

        json_loads = orjson.loads
        json_dumps = orjson_dumps
        arbitrary_types_allowed = True

    @validator("pose_actions", "observations")
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
        return f"{self.task}_{self.index}.json"

    def decompose(self) -> list[VIMAInstance]:
        """Decompose the instance into multiple instances.

        Each instance is the whole trajectory, whereas we don't want that.
        """
        logger.warning("Decomposing trajectories is not implemented yet. Just returning self.")
        return [self]
