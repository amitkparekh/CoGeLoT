from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Annotated

import numpy as np
import torch
from beartype.vale import Is
from pydantic import BaseModel
from pydantic_numpy import NDArray


if TYPE_CHECKING:
    from collections.abc import ItemsView, KeysView, ValuesView


class View(Enum):
    """Different views for the same image."""

    front = "front"
    top = "top"


class ImageType(Enum):
    """Different types of images."""

    rgb = "rgb"
    segmentation = "segm"
    depth = "depth"


class Bbox(BaseModel):
    """Bounding box."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int
    width: int
    height: int

    @classmethod
    def from_abs_xyxy(cls, x_min: int, x_max: int, y_min: int, y_max: int) -> Bbox:
        """Create from absolute XYXY coordinates."""
        return cls(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            width=x_max - x_min,
            height=y_max - y_min,
        )

    @classmethod
    def from_abs_xywh(cls, x_min: int, y_min: int, width: int, height: int) -> Bbox:
        """Create from absolute XYWH coordinates."""
        return cls(
            x_min=x_min,
            y_min=y_min,
            x_max=x_min + width,
            y_max=y_min + height,
            width=width,
            height=height,
        )

    @property
    def x_center(self) -> int:
        """Get the X center of the bounding box."""
        return (self.x_min + self.x_max) // 2

    @property
    def y_center(self) -> int:
        """Get the Y center of the bounding box."""
        return (self.y_min + self.y_max) // 2

    @property
    def as_xcychw(self) -> tuple[int, int, int, int]:
        """Return as a tuple of (xc, yc, h, w), which is what they use."""
        return (self.x_center, self.y_center, self.height, self.width)


PositionTensor = Annotated[
    torch.Tensor,
    Is[lambda tens: tens.numel() == 3 and tens.dtype is torch.float],  # noqa: PLR2004,WPS221
]
RotationTensor = Annotated[
    torch.Tensor,
    Is[lambda tens: tens.numel() == 4 and tens.dtype is torch.float],  # noqa: PLR2004,WPS221
]


class Position(BaseModel):
    """Position of a pose."""

    x: float
    y: float
    z: float

    @classmethod
    def from_tensor(cls, tensor: PositionTensor) -> Position:
        """Instantiate from a tensor."""
        flattened_tensor: list[float] = tensor.flatten().tolist()
        return cls(
            x=flattened_tensor[0],
            y=flattened_tensor[1],
            z=flattened_tensor[2],
        )

    @property
    def as_tensor(self) -> PositionTensor:
        """Convert the position to a tensor."""
        return torch.tensor([self.x, self.y, self.z])


class Rotation(BaseModel):
    """Rotation of a pose."""

    x: float
    y: float
    z: float
    w: float

    @classmethod
    def from_tensor(cls, tensor: RotationTensor) -> Rotation:
        """Instantiate from a tensor."""
        flattened_tensor: list[float] = tensor.flatten().tolist()
        return cls(
            x=flattened_tensor[0],
            y=flattened_tensor[1],
            z=flattened_tensor[2],
            w=flattened_tensor[3],
        )

    @property
    def as_tensor(self) -> RotationTensor:
        """Convert the rotation to a tensor."""
        return torch.tensor([self.x, self.y, self.z, self.w])


class Frame(BaseModel, arbitrary_types_allowed=True):
    """Get the output of a given modality for the various views."""

    front: NDArray
    top: NDArray

    def get_view(self, view: View) -> np.ndarray:
        """Get the perspective of the asset."""
        return getattr(self, view.value)


class Timestep(BaseModel):
    """Something that has an index to track time."""

    index: int


class Asset(BaseModel):
    """A single observation within the envirnment."""

    _obj_ids_to_ignore: set[int] = {0, 1}

    rgb: Frame
    segm: Frame

    @property
    def object_ids(self) -> set[int]:
        """Get the object IDs for the asset, given the current placeholder type.

        For some reason, they ignore {0,1} no matter what. So we do the same here.
        """
        get_unique_ids_from_segmentation = np.unique([self.segm.front, self.segm.top]).tolist()
        unique_ids = set(get_unique_ids_from_segmentation) - self._obj_ids_to_ignore
        return unique_ids


class Assets(BaseModel):
    """Structure to group all the assets."""

    __root__: dict[str, Asset]

    def __getitem__(self, item: str) -> Asset:  # noqa: WPS110
        """Let the Assets class be subscriptable like a dictionary."""
        return self.__root__[item]

    def __len__(self) -> int:
        """Get the number of assets."""
        return len(self.__root__)

    @property
    def all_object_ids(self) -> set[int]:
        """Get all the object IDs for all the assets."""
        all_object_ids: set[int] = set()
        for asset in self.values():
            all_object_ids.update(asset.object_ids)
        return all_object_ids

    def keys(self) -> KeysView[str]:
        """Get the keys of the assets."""
        return self.__root__.keys()

    def values(self) -> ValuesView[Asset]:  # noqa: WPS110
        """Get the values of the assets."""
        return self.__root__.values()

    def items(self) -> ItemsView[str, Asset]:  # noqa: WPS110
        """Get the items of the assets."""
        return self.__root__.items()

    def get_asset_names(self) -> list[str]:
        """Get all the asset names."""
        return list(self.__root__.keys())

    def get_asset_from_name(self, name: str) -> Asset:
        """Get the asset from the asset name."""
        # Ensure that the asset name is in the assets dict
        if name not in self.__root__:
            raise KeyError(f"Asset with name {name} not found!")
        return self[name]

    def get_asset_from_placeholder(self, placeholder: str) -> Asset:
        """Get the asset using the placeholder."""
        # Get the name of the asset by removing the left/right synbols
        asset_name = placeholder[1:-1]
        return self.get_asset_from_name(asset_name)


class Observation(Timestep, Asset):
    """A single observation within the envirnment."""

    def to_image_per_type_per_view(self) -> dict[View, dict[ImageType, np.ndarray]]:
        """Convert the observation to a dictionary of images per view."""
        return {
            view: {
                image_type: getattr(self, image_type.value).get_view(view)
                for image_type in (ImageType.rgb, ImageType.segmentation)
            }
            for view in View
        }

    def to_image_per_view_per_type(self) -> dict[ImageType, dict[View, np.ndarray]]:
        """Convert the observation to a dictionary of images per view."""
        return {
            image_type: {view: getattr(self, image_type.value).get_view(view) for view in View}
            for image_type in (ImageType.rgb, ImageType.segmentation)
        }


class Action(Timestep, arbitrary_types_allowed=True):
    """A single action taken by the agent in the environment."""
