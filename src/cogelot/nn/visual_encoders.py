from typing import ClassVar, Protocol, Self, TypeVar, cast

import torch

from cogelot.nn.shuffle_obj import shuffle_objects_for_each_observation
from vima.nn.obj_encoder import GatoMultiViewRGBEncoder, ObjEncoder
from vima.utils import DataDict

TheirEncoder = TypeVar("TheirEncoder", ObjEncoder, GatoMultiViewRGBEncoder)


class VisualEncoder(Protocol):
    """Protocol for the visual encoder."""

    is_patches: ClassVar[bool]

    # Used to ensure that we are converting the right encoder correctly.
    _parent_encoder: ClassVar[type[ObjEncoder] | type[GatoMultiViewRGBEncoder]]

    @classmethod
    def from_their_encoder(cls, encoder: TheirEncoder) -> Self:
        """Convert their encoder into the one we use.

        This is Python magic to avoid needing to re-implment the entire class. I like this magic.
        But the type hinting is a bit of a pain for it. There's a test to make sure this works.

        https://stackoverflow.com/a/8545134
        """
        assert issubclass(encoder.__class__, cls._parent_encoder)
        encoder.__class__ = type(cls.__name__, (cls, encoder.__class__), {})
        return cast(Self, encoder)

    def forward_observation(
        self, observation: DataDict, *, shuffle_obj_per_observation: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the visual observations."""
        ...

    def forward_prompt_visual(self, prompt_visual: DataDict) -> torch.Tensor:
        """Encode the prompt visual observations."""
        ...


class ObjectCentricVisualEncoder(ObjEncoder, VisualEncoder):
    """Wrapper on the VIMA object-centric visual encoder."""

    is_patches: ClassVar[bool] = False
    _parent_encoder = ObjEncoder

    def forward_observation(
        self, observation: DataDict, *, shuffle_obj_per_observation: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the object-centric observations."""
        obs_objects, ee = observation["objects"], observation["ee"]

        assert isinstance(obs_objects, DataDict)
        assert isinstance(ee, torch.Tensor)

        leading_dims = ee.shape[:2]

        if shuffle_obj_per_observation:
            obs_objects = shuffle_objects_for_each_observation(obs_objects)

        # Get the features for each image/obj/obs
        obs_objects = obs_objects.map_structure(
            func=lambda x: x.reshape(-1, *x.shape[3:]),
        )
        # I know the type from manual inspection
        obs_objects = cast(dict[str, dict[str, torch.Tensor]], obs_objects)
        img_feats = self(**obs_objects)
        img_feats = img_feats.reshape(*leading_dims, *img_feats.shape[1:])

        # Create mask for obs
        obj_mask = {
            obj_key: obs_objects["mask"][obj_key].reshape(*leading_dims, -1)
            for obj_key in obs_objects["mask"]
        }
        obj_mask_tensor = torch.cat(
            [obj_mask[view] for view in sorted(self._views)],
            dim=-1,
        )

        # Convert to the PyTorch-style mask, where True means it IS MASKED. The VIMA source opts
        # for the other approach, and we are going to be consistent dammit.
        obj_mask_tensor = ~obj_mask_tensor

        return img_feats, obj_mask_tensor

    def forward_prompt_visual(self, prompt_visual: DataDict) -> torch.Tensor:
        """Forward straight to the ObjEncoder."""
        return self.forward(
            cropped_img=prompt_visual["cropped_img"],
            bbox=prompt_visual["bbox"],
            mask=prompt_visual["mask"],
        )


class PatchesVisualEncoder(GatoMultiViewRGBEncoder, VisualEncoder):
    """Wrapper on the patch-based visual encoder."""

    is_patches: ClassVar[bool] = True
    _parent_encoder = GatoMultiViewRGBEncoder

    def forward_observation(
        self, observation: DataDict, *, shuffle_obj_per_observation: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the patches for the visual observations."""
        if shuffle_obj_per_observation:
            raise ValueError("Shuffling not supported for patch-based visual encoder.")

        rgbs, ee = observation["rgb"], observation["ee"]

        assert isinstance(rgbs, DataDict)
        assert isinstance(ee, torch.Tensor)

        leading_dims = ee.shape[:2]

        rgbs = rgbs.map_structure(
            func=lambda x: x.reshape(-1, *x.shape[2:]),
        )
        rgbs = cast(dict[str, torch.Tensor], rgbs)
        img_feats = self.forward(rgb=rgbs)
        img_feats = img_feats.reshape(*leading_dims, *img_feats.shape[1:])

        # Since it's patches, all the obs are used so the mask is all False
        # Also, this is pytorch-style masking
        observation_mask = torch.zeros(
            img_feats.shape[:-1], dtype=torch.bool, device=img_feats.device
        )

        return img_feats, observation_mask

    def forward_prompt_visual(self, prompt_visual: DataDict) -> torch.Tensor:
        """Forward straight to the ObjEncoder."""
        return self.forward(rgb=prompt_visual["rgb"])
