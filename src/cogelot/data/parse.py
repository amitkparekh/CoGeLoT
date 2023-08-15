import pickle
import shutil
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, get_args

import numpy as np
from PIL import Image

from cogelot.structures.common import Assets, Observation
from cogelot.structures.vima import (
    ActionBounds,
    ObjectMetadata,
    PoseAction,
    PoseActionType,
    Task,
    VIMAInstance,
)


POSE_ACTION_KEYS: tuple[PoseActionType, ...] = get_args(PoseActionType)
ACTIONS_FILE_NAME = "action.pkl"
TRAJECTORY_METADATA_FILE_NAME = "trajectory.pkl"
OBSERVATIONS_FILE_NAME = "obs.pkl"
RGB_PATH_PER_VIEW: Mapping[str, str] = MappingProxyType({"top": "rgb_top", "front": "rgb_front"})


def get_all_raw_instance_directories(raw_data_dir: Path) -> Iterator[Path]:
    """Get all the instance directories."""
    return raw_data_dir.glob("*/*/")


def load_rgb_observation_image(
    *, instance_dir: Path, view: Literal["top", "front"], frame_idx: int
) -> np.ndarray:
    """Load the RGB image of the observation for the given view."""
    image_path = instance_dir.joinpath(RGB_PATH_PER_VIEW[view], f"{frame_idx}.jpg")
    with Image.open(image_path) as image:
        # Also move the axes to be in the same structure as the prompt assets
        return np.ascontiguousarray(np.moveaxis(np.array(image), -1, 0))


def load_data_from_pickle(pickled_file: Path) -> Any:
    """Load the data from a pickle file."""
    return pickle.load(pickled_file.open("rb"))  # noqa: S301


def parse_object_metadata(trajectory_metadata: dict[str, Any]) -> list[ObjectMetadata]:
    """Extract the object metadata from the trajectory metadata."""
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


def parse_pose_actions(instance_dir: Path) -> list[PoseAction]:
    """Parse the pose actions.

    For the positions, they have removed the z-axis dimension from the data, so we are doing the
    same here.
    """
    raw_action_data = load_data_from_pickle(instance_dir.joinpath(ACTIONS_FILE_NAME))
    num_actions = len(raw_action_data[POSE_ACTION_KEYS[0]])
    actions_dict = {key: raw_action_data[key] for key in POSE_ACTION_KEYS}
    actions = [
        PoseAction(
            index=action_idx,
            pose0_position=actions_dict["pose0_position"][action_idx, :-1],
            pose1_position=actions_dict["pose1_position"][action_idx, :-1],
            pose0_rotation=actions_dict["pose0_rotation"][action_idx],
            pose1_rotation=actions_dict["pose1_rotation"][action_idx],
        )
        for action_idx in range(num_actions)
    ]
    return actions


def parse_observations(instance_dir: Path) -> list[Observation]:
    """Parse observations from raw data."""
    raw_obs_data = load_data_from_pickle(instance_dir.joinpath(OBSERVATIONS_FILE_NAME))
    raw_segmentation_data = raw_obs_data["segm"]
    num_obserations = len(raw_segmentation_data["top"])

    observations: list[Observation] = [
        Observation.parse_obj(
            {
                "index": obs_idx,
                "rgb": {
                    "front": load_rgb_observation_image(
                        instance_dir=instance_dir, view="front", frame_idx=obs_idx
                    ),
                    "top": load_rgb_observation_image(
                        instance_dir=instance_dir, view="top", frame_idx=obs_idx
                    ),
                },
                "segm": {
                    "front": raw_segmentation_data["front"][obs_idx],
                    "top": raw_segmentation_data["top"][obs_idx],
                },
            }
        )
        for obs_idx in range(num_obserations)
    ]

    return observations


def create_vima_instance_from_instance_dir(instance_dir: Path) -> VIMAInstance:
    """Create a VIMAInstance from their instance dir."""
    trajectory_metadata = load_data_from_pickle(
        instance_dir.joinpath(TRAJECTORY_METADATA_FILE_NAME)
    )

    pose_actions = parse_pose_actions(instance_dir)
    observations = parse_observations(instance_dir)

    # Each observation should be able to be paired with a pose action that was taken
    if len(observations) > len(pose_actions):
        observations = observations[: len(pose_actions)]

    if len(observations) != len(pose_actions):
        raise ValueError(
            f"Number of observations ({len(observations)}) does not match number of pose actions "
            f"({len(pose_actions)}) for instance {instance_dir}"
        )

    return VIMAInstance(
        index=int(instance_dir.stem),
        task=Task[instance_dir.parent.stem],
        total_steps=trajectory_metadata["steps"],
        prompt=trajectory_metadata["prompt"],
        prompt_assets=Assets.parse_obj(trajectory_metadata["prompt_assets"]),
        end_effector_type=trajectory_metadata["end_effector_type"],
        object_metadata=parse_object_metadata(trajectory_metadata),
        action_bounds=ActionBounds.parse_obj(trajectory_metadata["action_bounds"]),
        pose_actions=pose_actions,
        observations=observations,
    )


def parse_and_save_instance(
    raw_instance_dir: Path,
    *,
    output_dir: Path,
    delete_raw_instance_dir: bool = False,
    replace_if_exists: bool = False,
) -> None:
    """Parse the raw instance and save it to the output dir.

    Because there are repeated values within the JSON (as a result of the numpy arrays), each json
    is also compressed with gzip to make the file size smaller.

    Optionally, delete the raw instance dir if desired.
    """
    instance = create_vima_instance_from_instance_dir(raw_instance_dir)

    if not replace_if_exists and output_dir.joinpath(instance.file_name).exists():
        return

    instance.save(output_dir, compress=True)
    if delete_raw_instance_dir:
        shutil.rmtree(raw_instance_dir)
