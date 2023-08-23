import abc
from collections.abc import ItemsView, KeysView, ValuesView
from enum import Enum
from functools import cached_property
from typing import Any, Self

import datasets
import numpy as np
import torch
from numpy import typing as npt
from pydantic import BaseModel, ConfigDict, RootModel, field_validator


class PydanticHFDatasetMixin(abc.ABC):
    """Mixin for Pydantic models that will be used in/as/by HF datasets."""

    @classmethod
    @abc.abstractmethod
    def dataset_features(cls) -> datasets.Features:
        """Export the features schema for the HF dataset."""
        raise NotImplementedError


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
    def from_abs_xyxy(cls, x_min: int, x_max: int, y_min: int, y_max: int) -> Self:
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
    def from_abs_xywh(cls, x_min: int, y_min: int, width: int, height: int) -> Self:
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

    @classmethod
    def dataset_feature(cls) -> datasets.Sequence:
        """Feature for the HF dataset."""
        return datasets.Sequence(id="bbox", length=4, feature=datasets.Value("int32"))


class Position(BaseModel):
    """Position of a pose."""

    x: float
    y: float
    z: float

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> Self:
        """Instantiate from a tensor."""
        flattened_tensor: list[float] = tensor.flatten().tolist()
        return cls(
            x=flattened_tensor[0],
            y=flattened_tensor[1],
            z=flattened_tensor[2],
        )

    @property
    def as_tensor(self) -> torch.Tensor:
        """Convert the position to a tensor."""
        return torch.tensor([self.x, self.y, self.z])


class Rotation(BaseModel):
    """Rotation of a pose."""

    x: float
    y: float
    z: float
    w: float

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> Self:
        """Instantiate from a tensor."""
        flattened_tensor: list[float] = tensor.flatten().tolist()
        return cls(
            x=flattened_tensor[0],
            y=flattened_tensor[1],
            z=flattened_tensor[2],
            w=flattened_tensor[3],
        )

    @property
    def as_tensor(self) -> torch.Tensor:
        """Convert the rotation to a tensor."""
        return torch.tensor([self.x, self.y, self.z, self.w])


FRAME_SHAPE = (128, 256, 3)
NumpyImage = npt.NDArray[np.uint8]


class Frame(BaseModel, PydanticHFDatasetMixin):
    """Get the output of a given modality for the various views."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    front: NumpyImage
    top: NumpyImage

    def get_view(self, view: View) -> NumpyImage:
        """Get the perspective of the asset."""
        return getattr(self, view.value)

    @classmethod
    def dataset_features(cls) -> datasets.Features:
        """Export the features schema for the HF dataset."""
        return datasets.Features({"front": datasets.Image(), "top": datasets.Image()})


class Timestep(BaseModel, PydanticHFDatasetMixin):
    """Something that has an index to track time."""

    index: int

    @classmethod
    def dataset_features(cls) -> datasets.Features:
        """Export the features schema for the HF dataset."""
        return datasets.Features({"index": datasets.Value("int64")})


class Asset(BaseModel, PydanticHFDatasetMixin):
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

    @classmethod
    def dataset_features(cls) -> datasets.Features:
        """Export the features schema for the HF dataset."""
        return datasets.Features(
            {
                "rgb": Frame.dataset_features(),
                "segm": Frame.dataset_features(),
            }
        )


class PromptAsset(Asset):
    """A single prompt asset within the environment."""

    name: str

    @field_validator("rgb")
    @classmethod
    def check_shape_is_correct(cls, frame: Frame) -> Frame:
        """Ensure that the shape of the frame is correct."""
        if frame.front.shape != FRAME_SHAPE:
            frame.front = np.moveaxis(frame.front, 0, -1)

        if frame.top.shape != FRAME_SHAPE:
            frame.top = np.moveaxis(frame.top, 0, -1)

        assert frame.front.shape == FRAME_SHAPE
        assert frame.top.shape == FRAME_SHAPE
        return frame

    @classmethod
    def dataset_features(cls) -> datasets.Features:
        """Export the features schema for the HF dataset."""
        return datasets.Features(
            {
                "name": datasets.Value("string"),
                **Asset.dataset_features(),
            }
        )


class PromptAssets(RootModel[list[PromptAsset]]):
    """Structure to group all the assets."""

    root: list[PromptAsset]

    @classmethod
    def from_raw_prompt_assets(cls, raw_prompt_assets: dict[str, Any]) -> Self:
        """Instantiate from the raw trajectory metadata, from the environment."""
        return cls(
            root=[
                PromptAsset.model_validate({"name": asset_name, **asset_data})
                for asset_name, asset_data in raw_prompt_assets.items()
            ]
        )

    @cached_property
    def as_dict(self) -> dict[str, PromptAsset]:
        """Convert the assets to a dictionary."""
        return {asset.name: asset for asset in self.root}

    def __getitem__(self, item: str) -> PromptAsset:  # noqa: WPS110
        """Let the Assets class be subscriptable like a dictionary."""
        return self.as_dict[item]

    def __len__(self) -> int:
        """Get the number of assets."""
        return len(self.root)

    @property
    def all_object_ids(self) -> set[int]:
        """Get all the object IDs for all the assets."""
        all_object_ids: set[int] = set()
        for asset in self.values():
            all_object_ids.update(asset.object_ids)
        return all_object_ids

    def keys(self) -> KeysView[str]:
        """Get the keys of the assets."""
        return self.as_dict.keys()

    def values(self) -> ValuesView[PromptAsset]:  # noqa: WPS110
        """Get the values of the assets."""
        return self.as_dict.values()

    def items(self) -> ItemsView[str, PromptAsset]:  # noqa: WPS110
        """Get the items of the assets."""
        return self.as_dict.items()

    def get_asset_names(self) -> list[str]:
        """Get all the asset names."""
        return list(self.as_dict.keys())

    def get_asset_from_name(self, name: str) -> PromptAsset:
        """Get the asset from the asset name."""
        # Ensure that the asset name is in the assets dict
        if name not in self.as_dict:
            raise KeyError(f"Asset with name {name} not found!")
        return self[name]

    def get_asset_from_placeholder(self, placeholder: str) -> PromptAsset:
        """Get the asset using the placeholder."""
        # Get the name of the asset by removing the left/right synbols
        asset_name = placeholder[1:-1]
        return self.get_asset_from_name(asset_name)


class Observation(Timestep, Asset, PydanticHFDatasetMixin):
    """A single observation within the envirnment."""

    def to_image_per_type_per_view(self) -> dict[View, dict[ImageType, NumpyImage]]:
        """Convert the observation to a dictionary of images per view."""
        return {
            view: {
                image_type: getattr(self, image_type.value).get_view(view)
                for image_type in (ImageType.rgb, ImageType.segmentation)
            }
            for view in View
        }

    def to_image_per_view_per_type(self) -> dict[ImageType, dict[View, NumpyImage]]:
        """Convert the observation to a dictionary of images per view."""
        return {
            image_type: {view: getattr(self, image_type.value).get_view(view) for view in View}
            for image_type in (ImageType.rgb, ImageType.segmentation)
        }

    @classmethod
    def dataset_features(cls) -> datasets.Features:
        """Export the features schema for the HF dataset."""
        return datasets.Features(
            {
                **Timestep.dataset_features(),
                **Asset.dataset_features(),
            }
        )


class Action(Timestep):
    """A single action taken by the agent in the environment."""
