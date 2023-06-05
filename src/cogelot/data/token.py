from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from cogelot.data.structures import Bbox, BboxNumpy, ImageNumpy, Modality, View


class VisualObject(BaseModel, arbitrary_types_allowed=True):
    """A single object in the visual token."""

    bbox: Bbox
    cropped_image: ImageNumpy
    view: View


class Token(BaseModel):
    """A single token."""

    modality: Modality
    token: str
    position: int = Field(..., ge=0)


class TextToken(Token):
    """A single token with text information."""

    modality: Modality = Modality.TEXT
    token_id: int

    def __len__(self) -> int:
        """Since it is a text token, it just has a length of 1."""
        return 1


class VisualToken(Token):
    """A single token with visual information."""

    modality: Modality = Modality.IMAGE
    objects: list[VisualObject]

    def __len__(self) -> int:
        """Get the number of objects represented by the visual token."""
        return len(self.objects)

    def get_objects_for_view(self, view: View) -> list[VisualObject]:
        """Get all the objects for a given view."""
        return [obj for obj in self.objects if obj.view == view]

    def get_cropped_images_for_view(self, view: View) -> ImageNumpy:
        """Get all the cropped images per view."""
        objects_for_view = self.get_objects_for_view(view)
        all_cropped_images = [obj.cropped_image for obj in objects_for_view]
        cropped_images_array = np.asarray(all_cropped_images)
        return cropped_images_array

    def get_bounding_boxes_for_view(self, view: View) -> BboxNumpy:
        """Get all the bounding boxes per view."""
        objects_for_view = self.get_objects_for_view(view)
        all_bboxes = [obj.bbox.as_xcychw for obj in objects_for_view]
        bboxes_array = np.asarray(all_bboxes)
        return bboxes_array
