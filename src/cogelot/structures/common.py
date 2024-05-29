import abc
from collections.abc import ItemsView, Iterator, KeysView, ValuesView
from enum import Enum
from functools import cached_property
from typing import Annotated, Any, Self, TypeVar

import datasets
import numpy as np
import polars as pl
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from polars.type_aliases import SchemaDict
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    RootModel,
    field_validator,
)

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


FRAME_SHAPE: tuple[int, int, int] = (3, 128, 256)
PydanticTensor = Annotated[
    torch.Tensor,
    BeforeValidator(lambda tensor: torch.tensor(tensor) if isinstance(tensor, list) else tensor),
    BeforeValidator(
        lambda tensor: torch.from_numpy(tensor) if isinstance(tensor, np.ndarray) else tensor
    ),
    PlainSerializer(lambda tensor: tensor.cpu().numpy().tolist(), when_used="json"),
]


class ObjectDescription(BaseModel, PydanticHFDatasetMixin):
    """Simple structure of an object description metadata."""

    name: str
    texture: str

    @classmethod
    def dataset_features(cls) -> datasets.Features:
        """Export the features schema for the HF dataset."""
        return datasets.Features(
            {
                "name": datasets.Value("string"),
                "texture": datasets.Value("string"),
            }
        )

    def __eq__(self, value: object) -> bool:
        """Is this object description the same as another?"""
        if not isinstance(value, ObjectDescription):
            return False
        return self.name == value.name and self.texture == value.texture

    def __str__(self) -> str:
        """Convert the object description to a string."""
        return f"{self.texture} {self.name}"


class Frame(BaseModel, PydanticHFDatasetMixin):
    """Get the output of a given modality for the various views."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    front: PydanticTensor
    top: PydanticTensor

    def get_view(self, view: View) -> torch.Tensor:
        """Get the perspective of the asset."""
        return getattr(self, view.value)

    @field_validator("front", "top")
    @classmethod
    def check_shape_of_frame(cls, tensor: torch.Tensor) -> torch.Tensor:
        """Verify the shape of the frame."""
        expected_frame_shape = FRAME_SHAPE[-tensor.ndim :]
        if tensor.shape != expected_frame_shape:
            raise AssertionError(f"Expected shape {expected_frame_shape}, got {tensor.shape}")
        return tensor


class RGBFrame(Frame):
    """Frame for an RGB image."""

    @classmethod
    def dataset_features(cls) -> datasets.Features:
        """Export the features schema for the HF dataset."""
        array = datasets.Array3D(shape=FRAME_SHAPE, dtype="uint8")
        return datasets.Features({"front": array, "top": array})


class SegmentationFrame(Frame):
    """Frame for a segmentation image."""

    @classmethod
    def dataset_features(cls) -> datasets.Features:
        """Export the features schema for the HF dataset."""
        array = datasets.Array2D(shape=FRAME_SHAPE[1:], dtype="uint8")
        return datasets.Features({"front": array, "top": array})


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

    rgb: RGBFrame
    segm: SegmentationFrame

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
                "rgb": RGBFrame.dataset_features(),
                "segm": SegmentationFrame.dataset_features(),
            }
        )


class PromptAsset(Asset):
    """A single prompt asset within the environment."""

    name: str
    descriptions: Annotated[
        list[ObjectDescription],
        BeforeValidator(maybe_convert_dict_list_to_list_dict),
    ]
    # obj_name: str = ""
    # obj_color: str = ""

    @property
    def is_object_placholder(self) -> bool:
        """Is the placeholder an object placeholder?"""
        return len(self.descriptions) == 1

    @property
    def is_scene_placeholder(self) -> bool:
        """Is the placeholder a scene placeholder?"""
        return len(self.descriptions) > 1

    @classmethod
    def from_object_placeholder_type(cls, name: str, raw_asset_data: dict[str, Any]) -> Self:
        """Instantiate from a raw prompt asset."""
        return cls(
            name=name,
            descriptions=[
                ObjectDescription(
                    name=raw_asset_data["segm"]["obj_info"]["obj_name"],
                    texture=raw_asset_data["segm"]["obj_info"]["obj_color"],
                )
            ],
            **raw_asset_data,
        )

    @classmethod
    def from_scene_placeholder_type(cls, name: str, raw_asset_data: dict[str, Any]) -> Self:
        """Instantiate from a a raw scene placeholder type."""
        return cls(
            name=name,
            descriptions=[
                ObjectDescription(
                    name=obj_info["obj_name"],
                    texture=obj_info["obj_color"],
                )
                for obj_info in raw_asset_data["segm"]["obj_info"]
            ],
            **raw_asset_data,
        )

    @classmethod
    def dataset_features(cls) -> datasets.Features:
        """Export the features schema for the HF dataset."""
        return datasets.Features(
            {
                "name": datasets.Value("string"),
                "descriptions": datasets.Sequence(ObjectDescription.dataset_features()),
                **Asset.dataset_features(),
            }
        )

    @property
    def object_description(self) -> ObjectDescription | None:
        """Get the object description."""
        if self.is_scene_placeholder:
            return None
        return self.descriptions[0]

    @property
    def as_natural_language(self) -> str | None:
        """Convert the properties to natural language."""
        if self.object_description:
            return None
        return str(self.object_description)


class PromptAssets(RootModel[list[PromptAsset]]):
    """Structure to group all the assets."""

    root: list[PromptAsset]

    @classmethod
    def from_raw_prompt_assets(cls, raw_prompt_assets: dict[str, Any]) -> Self:
        """Instantiate from the raw trajectory metadata, from the environment."""
        prompt_assets = [
            PromptAsset.from_object_placeholder_type(asset_name, asset_data)
            if asset_data["placeholder_type"] == "object"
            else PromptAsset.from_scene_placeholder_type(asset_name, asset_data)
            for asset_name, asset_data in raw_prompt_assets.items()
        ]
        return cls(root=prompt_assets)

    @cached_property
    def as_dict(self) -> dict[str, PromptAsset]:
        """Convert the assets to a dictionary."""
        return {asset.name: asset for asset in self.root}

    def as_python_dict(self) -> dict[str, dict[str, Any]]:
        """Convert the assets to a python dictionary."""
        return {asset.name: asset.model_dump() for asset in self.root}

    def __getitem__(self, item: str) -> PromptAsset:
        """Let the Assets class be subscriptable like a dictionary."""
        return self.as_dict[item]

    def __len__(self) -> int:
        """Get the number of assets."""
        return len(self.root)

    @property
    def all_object_ids(self) -> set[int]:
        """Get all the object IDs for all the assets."""
        return set.union(*(asset.object_ids for asset in self.root))

    def keys(self) -> KeysView[str]:
        """Get the keys of the assets."""
        return self.as_dict.keys()

    def values(self) -> ValuesView[PromptAsset]:
        """Get the values of the assets."""
        return self.as_dict.values()

    def items(self) -> ItemsView[str, PromptAsset]:
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

    def to_image_per_type_per_view(self) -> dict[View, dict[ImageType, torch.Tensor]]:
        """Convert the observation to a dictionary of images per view."""
        return {
            view: {
                image_type: getattr(self, image_type.value).get_view(view)
                for image_type in (ImageType.rgb, ImageType.segmentation)
            }
            for view in View
        }

    def to_image_per_view_per_type(self) -> dict[ImageType, dict[View, torch.Tensor]]:
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


class ObservationVideos(BaseModel):
    """Observation videos for a single episode."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    front_rgb: torch.Tensor
    front_segm: torch.Tensor
    top_rgb: torch.Tensor
    top_segm: torch.Tensor

    def as_python(self) -> dict[str, list[Any]]:
        """Convert videos from tensors to python lists."""
        return {
            "front_rgb": self.front_rgb.cpu().numpy().tolist(),
            "front_segm": self.front_segm.cpu().numpy().tolist(),
            "top_rgb": self.top_rgb.cpu().numpy().tolist(),
            "top_segm": self.top_segm.cpu().numpy().tolist(),
        }

    @classmethod
    def polars_schema_override(cls) -> SchemaDict:
        """Get the schema override for the polars DataFrame."""
        observation_dtype = pl.List(
            pl.Array(pl.Array(pl.Array(pl.UInt8, 256), 128), 3),
        )
        return {
            "front_rgb": observation_dtype,
            "front_segm": observation_dtype,
            "top_rgb": observation_dtype,
            "top_segm": observation_dtype,
        }


class Observations(RootModel[list[Observation]]):
    """A sequence of observations in an environment."""

    root: list[Observation] = Field(default_factory=list)

    def __len__(self) -> int:
        """Get the number of assets."""
        return len(self.root)

    def __iter__(self) -> Iterator[Observation]:
        """Iterate over the observations."""
        return iter(self.root)

    def __getitem__(self, index: int) -> Observation:
        """Get the observation at the given index."""
        return self.root[index]

    @field_validator("root")
    @classmethod
    def sort_by_index(cls, indexed_steps: list[Timestep]) -> list[Timestep]:
        """Sort the steps by index."""
        indexed_steps.sort(key=lambda step: step.index)
        return indexed_steps

    def convert_to_videos(self) -> ObservationVideos:
        """Extract multiple videos from a list of observations."""
        rgb_front_frames = []
        segm_front_frames = []
        rgb_top_frames = []
        segm_top_frames = []

        for observation in self.root:
            obs = observation.to_image_per_type_per_view()
            rgb_front_frames.append(obs[View.front][ImageType.rgb])
            segm_front_frames.append(obs[View.front][ImageType.segmentation])
            rgb_top_frames.append(obs[View.top][ImageType.rgb])
            segm_top_frames.append(obs[View.top][ImageType.segmentation])

        front_segmentation = torch.stack(segm_front_frames, dim=0).long()
        top_segmentation = torch.stack(segm_top_frames, dim=0).long()
        colored_front_segmentation = self.segmentation_color_map()[front_segmentation]
        colored_top_segmentation = self.segmentation_color_map()[top_segmentation]

        return ObservationVideos(
            front_rgb=rearrange(rgb_front_frames, "t c h w -> t c h w"),
            top_rgb=rearrange(rgb_top_frames, "t c h w -> t c h w"),
            front_segm=rearrange(colored_front_segmentation, "t h w c -> t c h w"),
            top_segm=rearrange(colored_top_segmentation, "t h w c -> t c h w"),
        )

    @classmethod
    def segmentation_color_map(cls) -> torch.Tensor:
        """Get the segmentation color map."""
        color_map = plt.cm.tab20(range(20))  # pyright: ignore[reportAttributeAccessIssue]
        as_tensor = torch.tensor(color_map)[:, :-1]
        as_int_tensor = (as_tensor * 255).to(torch.uint8)
        return as_int_tensor


class Action(Timestep):
    """A single action taken by the agent in the environment."""
