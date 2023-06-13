from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
import orjson
from PIL import Image
from pydantic import BaseModel, root_validator

from cogelot.data.constants import EndEffector


def orjson_dumps(v: Any, *, default: Any) -> str:
    """Convert Model to JSON string.

    orjson.dumps returns bytes, to match standard json.dumps we need to decode.
    """
    return orjson.dumps(
        v,
        default=default,
        option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_APPEND_NEWLINE,
    ).decode()


class ObjectMetadata(BaseModel):
    """Metadata for a given object."""

    obj_id: int
    obj_name: str
    obj_asset_name: str
    texture_name: str


class PoseActions(BaseModel, arbitrary_types_allowed=True):
    """Actions which are taken by the agent in the environment."""

    pose0_position: np.ndarray
    pose1_position: np.ndarray
    pose0_rotation: np.ndarray
    pose1_rotation: np.ndarray

    @root_validator
    @classmethod
    def check_first_dim_is_identical(
        cls, values: dict[str, np.ndarray]  # noqa: WPS110
    ) -> dict[str, np.ndarray]:
        """Ensure that the first dims are identical."""
        first_dims = [v.shape[0] for v in values.values()]

        if not all(first_dims[0] == d for d in first_dims):
            raise ValueError("All first dims must be identical")

        return values

    def __len__(self) -> int:
        """Return the number of actions."""
        return self.pose0_position.shape[0]


class Observations(BaseModel, arbitrary_types_allowed=True):
    """Observations from the environment."""

    ee: np.ndarray
    rgb: dict[Literal["top", "front"], np.ndarray]
    segm: dict[Literal["top", "front"], np.ndarray]

    def __len__(self) -> int:
        """Return the number of observations."""
        return self.ee.shape[0]

    @root_validator
    @classmethod
    def check_first_dim_is_identical(
        cls, values: dict[str, Any]  # noqa: WPS110
    ) -> dict[str, Any]:
        """Ensure that the first dims are identical."""
        rgb_front = values["rgb"]["front"]
        rgb_top = values["rgb"]["top"]
        segm_front = values["segm"]["front"]
        segm_top = values["segm"]["top"]

        all_first_dims_are_identical = all(
            rgb_front.shape[0] == dim
            for dim in (rgb_top.shape[0], segm_front.shape[0], segm_top.shape[0])
        )

        if not all_first_dims_are_identical:
            raise ValueError("All first dims must be identical")

        return values


class VIMAInstance(BaseModel):
    """A single instance of the VIMA dataset, merging all the files into a single object."""

    index: int
    task: str
    path: Path

    object_metadata: list[ObjectMetadata]

    end_effector_type: EndEffector

    pose_actions: PoseActions
    observations: Observations

    prompt: str
    prompt_assets: dict[str, dict[str, Any]]

    class Config:
        """Pydantic config."""

        json_loads = orjson.loads
        json_dumps = orjson_dumps
        arbitrary_types_allowed = True

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
    def object_ids(self) -> list[int]:
        """Get the object ids."""
        return [obj.obj_id for obj in self.object_metadata]


class VIMAInstanceFactory:
    """Build VIMA instances from the raw data."""

    pose_action_keys = ("pose0_position", "pose1_position", "pose0_rotation", "pose1_rotation")
    actions_file_name = "action.pkl"
    trajectory_metadata_file_name = "trajectory.pkl"
    observations_file_name = "obs.pkl"
    rgb_path_per_view = {"top": "rgb_top", "front": "rgb_front"}

    def parse_from_instance_dir(self, instance_dir: Path) -> VIMAInstance:
        """Parse a VIMA instance from their instance dir."""
        trajectory_metadata = self._load_data_from_pickle(
            instance_dir.joinpath(self.trajectory_metadata_file_name)
        )

        return VIMAInstance(
            index=int(instance_dir.stem),
            path=instance_dir,
            task=instance_dir.parent.stem,
            prompt=trajectory_metadata["prompt"],
            prompt_assets=trajectory_metadata["prompt_assets"],
            end_effector_type=trajectory_metadata["end_effector_type"],
            object_metadata=self.parse_object_metadata(trajectory_metadata),
            pose_actions=self.parse_pose_actions(instance_dir),
            observations=self.parse_observations(instance_dir),
        )

    def parse_object_metadata(self, trajectory_metadata: dict[str, Any]) -> list[ObjectMetadata]:
        """Extract the object metadata."""
        object_metadata: list[ObjectMetadata] = []

        for object_id, object_info in trajectory_metadata["obj_id_to_info"].items():
            object_metadata.append(
                ObjectMetadata(
                    obj_id=object_id,
                    obj_name=object_info["obj_name"],
                    obj_asset_name=object_info["obj_assets"],
                    texture_name=object_info["texture_name"],
                )
            )

        return object_metadata

    def parse_pose_actions(self, instance_dir: Path) -> PoseActions:
        """Parse the pose actions."""
        actions = self._load_data_from_pickle(instance_dir.joinpath(self.actions_file_name))
        pose_actions = {key: actions[key] for key in self.pose_action_keys}

        return PoseActions.parse_obj(pose_actions)

    def parse_observations(self, instance_dir: Path) -> Observations:
        """Parse the observations."""
        observations = self._load_data_from_pickle(
            instance_dir.joinpath(self.observations_file_name)
        )
        rgb: dict[Literal["front", "top"], np.ndarray] = {
            "front": self._load_all_rgb_images(instance_dir=instance_dir, view="front"),
            "top": self._load_all_rgb_images(instance_dir=instance_dir, view="top"),
        }

        return Observations(ee=observations["ee"], rgb=rgb, segm=observations["segm"])

    def _load_data_from_pickle(self, pickled_file: Path) -> Any:
        """Load the data from a pickle file."""
        return pickle.load(pickled_file.open("rb"))  # noqa: S301

    def _load_all_rgb_images(
        self, *, instance_dir: Path, view: Literal["front", "top"]
    ) -> np.ndarray:
        """Load all the RGB images from the given dir."""
        images_dir = instance_dir.joinpath(self.rgb_path_per_view[view])
        images: list[np.ndarray] = []

        for image_path in sorted(images_dir.glob("*.jpg")):
            with Image.open(image_path) as image:
                images.append(np.array(image))

        return np.array(images)
