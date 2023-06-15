from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch
from pydantic import BaseModel, Field

from cogelot.structures.common import Bbox, PositionTensor, RotationTensor, View


if TYPE_CHECKING:
    from cogelot.structures.vima import PoseAction


class TokenType(Enum):
    """Different modalities that can be encoded."""

    text = "text"
    image = "image"
    end_effector = "end_effector"
    observation = "observation"
    action = "action"


class VisualObject(BaseModel, arbitrary_types_allowed=True):
    """A single object in the visual token."""

    bbox: Bbox
    cropped_image: torch.Tensor
    view: View


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


class ImageToken(Token):
    """A single token with visual information."""

    token_type: TokenType = TokenType.image
    objects: list[VisualObject]

    def get_objects_for_view(self, view: View) -> list[VisualObject]:
        """Get all the objects for a given view."""
        return [obj for obj in self.objects if obj.view == view]

    def get_cropped_images_for_view(self, view: View) -> torch.Tensor:
        """Get all the cropped images per view."""
        objects_for_view = self.get_objects_for_view(view)
        all_cropped_images = [obj.cropped_image for obj in objects_for_view]
        cropped_images_array = torch.stack(all_cropped_images, dim=0)
        return cropped_images_array

    def get_bounding_boxes_for_view(self, view: View) -> torch.Tensor:
        """Get all the bounding boxes per view."""
        objects_for_view = self.get_objects_for_view(view)
        all_bboxes = [obj.bbox.as_xcychw for obj in objects_for_view]
        bboxes_array = torch.tensor(all_bboxes)
        return bboxes_array

    def __len__(self) -> int:
        """Get the number of objects represented by the visual token."""
        return len(self.objects)


class EndEffectorToken(TextToken):
    """Token for the end effector."""

    token_type: TokenType = TokenType.end_effector


class ObservationToken(EndEffectorToken, ImageToken):
    """Token for an observation with end effector."""

    token_type: TokenType = TokenType.observation

    @classmethod
    def from_tokens(
        cls, *, image_token: ImageToken, end_effector_token: EndEffectorToken
    ) -> ObservationToken:
        """Instantiate by merging an image token and an end effector token."""
        return cls(
            index=image_token.index,
            token=image_token.token,
            objects=image_token.objects,
            token_id=end_effector_token.token_id,
        )


class PoseActionToken(Token, arbitrary_types_allowed=True):
    """Token for the agent action."""

    token_type: TokenType = TokenType.action

    pose0_position: PositionTensor
    pose1_position: PositionTensor
    pose0_rotation: RotationTensor
    pose1_rotation: RotationTensor

    @classmethod
    def from_pose_action(cls, pose_actions: PoseAction) -> PoseActionToken:
        """Create an action token from a pose action."""
        return PoseActionToken(
            index=pose_actions.index,
            pose0_position=torch.from_numpy(pose_actions.pose0_position),
            pose1_position=torch.from_numpy(pose_actions.pose1_position),
            pose0_rotation=torch.from_numpy(pose_actions.pose0_rotation),
            pose1_rotation=torch.from_numpy(pose_actions.pose1_rotation),
        )
