from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from cogelot.data.structures import Bbox, BboxNumpy, ImageNumpy, Modality, Perspective


class VisualObject(BaseModel):
    """A single object in the visual token."""

    bbox: Bbox
    cropped_image: ImageNumpy
    perspective: Perspective


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

    def get_objects_for_perspective(self, perspective: Perspective) -> list[VisualObject]:
        """Get all the objects for a given perspective."""
        return [obj for obj in self.objects if obj.perspective == perspective]

    def get_cropped_images_for_perspective(self, perspective: Perspective) -> ImageNumpy:
        """Get all the cropped images per perspective."""
        objects_for_perspective = self.get_objects_for_perspective(perspective)
        all_cropped_images = [obj.cropped_image for obj in objects_for_perspective]
        cropped_images_array = np.asarray(all_cropped_images)
        return cropped_images_array

    def get_bounding_boxes_for_perspective(self, perspective: Perspective) -> BboxNumpy:
        """Get all the bounding boxes per perspective."""
        objects_for_perspective = self.get_objects_for_perspective(perspective)
        all_bboxes = [obj.bbox.as_xcychw for obj in objects_for_perspective]
        bboxes_array = np.asarray(all_bboxes)
        return bboxes_array
