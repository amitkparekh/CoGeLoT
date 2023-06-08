from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from cogelot.data.structures import (
    Bbox,
    BboxNumpy,
    ImageNumpy,
    PoseAction,
    PositionTensor,
    RotationTensor,
    View,
)


class VisualObject(BaseModel, arbitrary_types_allowed=True):
    """A single object in the visual token."""

    bbox: Bbox
    cropped_image: ImageNumpy
    view: View


class TokenType(Enum):
    """Different modalities that can be encoded."""

    text = 0
    image = 1
    end_effector = 2
    action = 3


class Token(BaseModel):
    """A single token."""

    token_type: TokenType
    index: int = Field(..., ge=0)
    token: str | None = None

    def __len__(self) -> int:
        """Default token length to just 1."""
        return 1


class TextToken(Token):
    """A single token with text information."""

    token_type: TokenType = TokenType.text
    token_id: int


class VisualToken(Token):
    """A single token with visual information."""

    token_type: TokenType = TokenType.image
    objects: list[VisualObject]

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

    def __len__(self) -> int:
        """Get the number of objects represented by the visual token."""
        return len(self.objects)


class EndEffectorToken(TextToken):
    """Token for the end effector."""

    token_type: TokenType = TokenType.end_effector


class ActionToken(Token, arbitrary_types_allowed=True):
    """Token for the agent action."""

    token_type: TokenType = TokenType.action

    pose0_position: PositionTensor
    pose1_position: PositionTensor
    pose0_rotation: RotationTensor
    pose1_rotation: RotationTensor

    @classmethod
    def from_pose_action(cls, pose_action: PoseAction) -> ActionToken:
        """Create an action token from a pose action."""
        return ActionToken(
            index=pose_action.index,
            pose0_position=pose_action.pose0_position.as_tensor,
            pose1_position=pose_action.pose1_position.as_tensor,
            pose0_rotation=pose_action.pose0_rotation.as_tensor,
            pose1_rotation=pose_action.pose1_rotation.as_tensor,
        )
