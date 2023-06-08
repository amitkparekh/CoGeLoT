from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
import orjson
import torch
from PIL import Image
from pydantic import BaseModel, validator

from cogelot.data.constants import EndEffector
from cogelot.data.structures import (
    Assets,
    ObjectMetadata,
    Observation,
    PoseAction,
    Position,
    Rotation,
)


def orjson_dumps(v: Any, *, default: Any) -> str:
    """Convert Model to JSON string.

    orjson.dumps returns bytes, to match standard json.dumps we need to decode.
    """
    return orjson.dumps(
        v,
        default=default,
        option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_APPEND_NEWLINE,
    ).decode()


class VIMAInstance(BaseModel):
    """Parsed VIMA instance."""

    index: int
    path: Path
    task: str

    object_metadata: list[ObjectMetadata]

    end_effector: EndEffector

    prompt: str
    prompt_assets: Assets

    observations: list[Observation]
    actions: list[PoseAction]

    class Config:
        """Pydantic config."""

        json_loads = orjson.loads
        json_dumps = orjson_dumps

    @validator("observations")
    @classmethod
    def sort_obsevations(cls, observations: list[Observation]) -> list[Observation]:
        """Sort observations by their index."""
        return sorted(observations, key=lambda obs: obs.index)

    @validator("actions")
    @classmethod
    def sort_actions(cls, actions: list[PoseAction]) -> list[PoseAction]:
        """Sort actions by their index."""
        return sorted(actions, key=lambda action: action.index)

    @property
    def num_steps(self) -> int:
        """Get the number of steps in the instance."""
        return len(self.observations)

    @property
    def action_per_observation(self) -> dict[Observation, PoseAction | None]:
        """Get the action per observation."""
        actions = self.actions
        if len(self.observations) != len(self.actions):
            actions = [*actions, None]

        return dict(zip(self.observations, actions, strict=True))

    @property
    def num_objects(self) -> int:
        """Get the number of objects in the instance."""
        return len(self.object_metadata)

    @property
    def object_ids(self) -> list[int]:
        """Get the object ids."""
        return [obj.obj_id for obj in self.object_metadata]

    @overload
    def get_action_for_observation(self, observation: int) -> PoseAction | None:
        ...  # noqa: WPS428

    @overload
    def get_action_for_observation(self, observation: Observation) -> PoseAction | None:
        ...  # noqa: WPS428

    def get_action_for_observation(self, observation: int | Observation) -> PoseAction | None:
        """Get the action for a given observation.

        Args:
            observation: The observation index or the observation itself.
        """
        if isinstance(observation, int):
            observation = self.observations[observation]

        return self.action_per_observation[observation]


class VIMAInstanceFactory:
    """Build VIMA instances from their raw data."""

    pose_action_keys = ("pose0_position", "pose1_position", "pose0_rotation", "pose1_rotation")
    actions_file_name = "action.pkl"
    trajectory_metadata_file_name = "trajectory.pkl"
    observations_file_name = "obs.pkl"
    rgb_path_per_view = {"top": "rgb_top", "front": "rgb_front"}

    def parse_from_instance_dir(self, instance_dir: Path) -> VIMAInstance:
        """Parse a VIMA Instance from one of their instance dirs."""
        trajectory_metadata = self._load_trajectory_metadata(instance_dir)
        pose_actions = self.parse_pose_actions(instance_dir)
        observations = self.parse_observations(instance_dir)

        return VIMAInstance(
            index=int(instance_dir.stem),
            path=instance_dir,
            task=instance_dir.parent.stem,
            object_metadata=self._parse_object_metadata(trajectory_metadata),
            prompt=trajectory_metadata["prompt"],
            prompt_assets=Assets.parse_obj(trajectory_metadata["prompt_assets"]),
            end_effector=trajectory_metadata["end_effector_type"],
            actions=pose_actions,
            observations=observations,
        )

    def parse_observations(self, instance_dir: Path) -> list[Observation]:
        """Parse observations from raw data."""
        raw_obs_data = self._load_data_from_pickle(
            instance_dir.joinpath(self.observations_file_name)
        )
        raw_segmentation_data = raw_obs_data["segm"]

        num_obserations = len(raw_segmentation_data["top"])

        observations: list[Observation] = []

        for obs_idx in range(num_obserations):
            observations.append(
                Observation.parse_obj(
                    {
                        "index": obs_idx,
                        "rgb": {
                            "front": self._load_rgb_observation_image(
                                instance_dir=instance_dir, view="front", frame_idx=obs_idx
                            ),
                            "top": self._load_rgb_observation_image(
                                instance_dir=instance_dir, view="top", frame_idx=obs_idx
                            ),
                        },
                        "segm": {
                            "front": raw_segmentation_data["front"][obs_idx],
                            "top": raw_segmentation_data["top"][obs_idx],
                        },
                    }
                )
            )

        return observations

    def parse_pose_actions(self, instance_dir: Path) -> list[PoseAction]:
        """Parse pose actions from raw data."""
        raw_action_data = self._load_raw_pose_action_data(instance_dir)
        return self._parse_pose_actions_from_raw_data(raw_action_data)

    def _parse_object_metadata(self, trajectory_metadata: dict[str, Any]) -> list[ObjectMetadata]:
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

    def _load_data_from_pickle(self, pickled_file: Path) -> Any:
        """Load the data from a pickle file."""
        return pickle.load(pickled_file.open("rb"))  # noqa: S301

    def _load_trajectory_metadata(self, instance_dir: Path) -> dict[str, Any]:
        """Load the trajectory metadata."""
        return self._load_data_from_pickle(
            instance_dir.joinpath(self.trajectory_metadata_file_name)
        )

    def _parse_object_ids_and_labels(self, trajectory_metadata: dict[str, Any]) -> dict[int, str]:
        """Extract the object IDs and their labels."""
        return {
            obj_id: obj_info["obj_name"]
            for obj_id, obj_info in trajectory_metadata["obj_id_to_info"].items()
        }

    def _load_rgb_observation_image(
        self, *, instance_dir: Path, view: Literal["top", "front"], frame_idx: int
    ) -> np.ndarray:
        """Load the RGB image of the observation for the given view."""
        image_path = instance_dir.joinpath(self.rgb_path_per_view[view], f"{frame_idx}.jpg")
        with Image.open(image_path) as image:
            return np.array(image)

    def _get_num_actions_from_raw_pose_action_data(self, raw_action_data: dict[str, Any]) -> int:
        """Get the number of actions from the raw pose action data.

        All the pose actions should have the same batch size/identiacal first dimension
        """
        return len(raw_action_data[self.pose_action_keys[0]])

    def _parse_pose_actions_from_raw_data(
        self, raw_action_data: dict[str, Any]
    ) -> list[PoseAction]:
        """Parse pose actions from raw data."""
        tensors = {key: torch.tensor(raw_action_data[key]) for key in self.pose_action_keys}

        num_actions = self._get_num_actions_from_raw_pose_action_data(raw_action_data)

        actions = [
            PoseAction(
                index=action_idx,
                pose0_position=Position.from_tensor(tensors["pose0_position"][action_idx]),
                pose1_position=Position.from_tensor(tensors["pose1_position"][action_idx]),
                pose0_rotation=Rotation.from_tensor(tensors["pose0_rotation"][action_idx]),
                pose1_rotation=Rotation.from_tensor(tensors["pose1_rotation"][action_idx]),
            )
            for action_idx in range(num_actions)
        ]

        return actions

    def _load_raw_pose_action_data(self, instance_dir: Path) -> dict[str, Any]:
        """Load the raw pose action data."""
        return self._load_data_from_pickle(instance_dir.joinpath(self.actions_file_name))
