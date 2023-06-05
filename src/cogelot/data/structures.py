from __future__ import annotations

from enum import Enum

import numpy as np
from numpy import typing as npt
from pydantic import BaseModel, validator


ImageNumpy = npt.NDArray[np.int_]
BboxNumpy = npt.NDArray[np.int_]


class Modality(Enum):
    """Different modalities that can be encoded."""

    TEXT = 0
    IMAGE = 1


class View(Enum):
    """Different views for the same image."""

    FRONT = "front"
    TOP = "top"


class ImageType(Enum):
    """Different types of images."""

    RGB = "rgb"
    SEGMENTATION = "segmentation"
    DEPTH = "depth"


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


class ObjectInfo(BaseModel):
    """Metadata for a given object."""

    obj_id: int


class ImageView(BaseModel, arbitrary_types_allowed=True):
    """Get the output of a given modality for the various views."""

    front: ImageNumpy
    top: ImageNumpy

    def get_view(self, view: View) -> ImageNumpy:
        """Get the perspective of the asset."""
        return getattr(self, view.value)


class SegmentationModalityView(ImageView):
    """Various views for the segmentation modality.

    This explicitly also includes various information about each object in the view too.
    """

    obj_info: list[ObjectInfo]

    @validator("obj_info")
    @classmethod
    def ensure_obj_info_is_a_list(
        cls, obj_info: ObjectInfo | list[ObjectInfo]
    ) -> list[ObjectInfo]:
        """Ensure the object info is always a list."""
        if not isinstance(obj_info, list):
            return [obj_info]
        return obj_info

    @property
    def object_ids(self) -> list[int]:
        """Get the object ids from the object info."""
        return [info.obj_id for info in self.obj_info]


class Asset(BaseModel):
    """An asset within the environment."""

    rgb: ImageView
    segm: SegmentationModalityView

    @property
    def object_ids(self) -> list[int]:
        """Get the object IDs for the asset, given the current placeholder type."""
        return self.segm.object_ids


class Assets(BaseModel):
    """Structure to group all the assets."""

    __root__: dict[str, Asset]

    def __getitem__(self, item: str) -> Asset:
        """Let the Assets class be subscriptable like a dictionary."""
        return self.__root__[item]

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
