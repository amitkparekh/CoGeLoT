from __future__ import annotations

import cv2
import numpy as np
from einops import rearrange

from cogelot.data.structures import Bbox, ImageNumpy, ImageType, Perspective
from cogelot.data.token import VisualObject, VisualToken


class ImageTokenizer:
    """Tokenize images into visual tokens."""

    _minimum_dimension_size: int = 2

    def __init__(self, image_size: int = 32) -> None:
        self.image_size = image_size

    def create_visual_token_from_images(
        self,
        *,
        token_position_idx: int,
        token_value: str,
        image_per_type_per_perspective: dict[Perspective, dict[ImageType, ImageNumpy]],
        available_object_ids: list[int],
    ) -> VisualToken:
        """Create a visual token from the raw images."""
        all_visual_objects = self.extract_objects_from_images(
            image_per_type_per_perspective=image_per_type_per_perspective,
            available_object_ids=available_object_ids,
        )
        visual_token = self.create_visual_token_from_objects(
            token_position_idx=token_position_idx,
            token_value=token_value,
            objects=all_visual_objects,
        )
        return visual_token

    def create_visual_token_from_objects(
        self, *, token_position_idx: int, token_value: str, objects: list[VisualObject]
    ) -> VisualToken:
        """Create a visual token."""
        return VisualToken(token=token_value, position=token_position_idx, objects=objects)

    def extract_objects_from_images(
        self,
        *,
        image_per_type_per_perspective: dict[Perspective, dict[ImageType, ImageNumpy]],
        available_object_ids: list[int],
    ) -> list[VisualObject]:
        """Extract objects from images and create a visual token for the given perspective."""
        visual_objects: list[VisualObject] = []

        for perspective, images_per_type in image_per_type_per_perspective.items():
            for object_id in available_object_ids:
                # Extract the bounding box for the object ID from the segmentation image
                try:
                    bbox = self.create_bounding_box_from_segmentation_image(
                        object_id, images_per_type[ImageType.SEGMENTATION]
                    )
                except ValueError:
                    # If the object is too small, then we skip it
                    continue

                # Crop the object from the RGB image using the bounding box
                cropped_image = self.extract_object_from_image(
                    images_per_type[ImageType.RGB], bbox
                )

                # Create the visual object
                visual_objects.append(
                    VisualObject(perspective=perspective, bbox=bbox, cropped_image=cropped_image)
                )

        return visual_objects

    def create_bounding_box_from_segmentation_image(
        self, object_id: int, segmentation_image: ImageNumpy
    ) -> Bbox:
        """Extract the bounding box for the object ID from the segmentation view."""
        # Get all the pixels that belong to the object
        ys, xs = np.nonzero(segmentation_image == object_id)

        # Make sure that the object is not too small to be cropped
        if len(xs) < self._minimum_dimension_size or len(ys) < self._minimum_dimension_size:
            raise ValueError("The object is too small to be cropped!")

        # Create the bounding box
        bbox = Bbox.from_abs_xyxy(
            x_min=int(np.min(xs)),
            x_max=int(np.max(xs)),
            y_min=int(np.min(ys)),
            y_max=int(np.max(ys)),
        )
        return bbox

    def extract_object_from_image(self, image: ImageNumpy, bbox: Bbox) -> ImageNumpy:
        """Crop the object from the RGB image using the bounding box."""
        cropped_image = self.crop_to_bounding_box(image, bbox)

        # Make sure the cropped image is square
        cropped_image = self.pad_to_square(cropped_image)

        # Resize the image
        cropped_image = self.resize(cropped_image)

        return cropped_image

    def resize(self, image: ImageNumpy) -> ImageNumpy:
        """Resize the image."""
        image = rearrange(image, "c h w -> h w c")
        image = np.asarray(image)
        image = cv2.resize(
            image,  # pyright: ignore[reportGeneralTypeIssues]
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        image = rearrange(image, "h w c -> c h w")
        image = np.asarray(image)
        return image

    def crop_to_bounding_box(self, image: ImageNumpy, bbox: Bbox) -> ImageNumpy:
        """Crop the image to the bounding box."""
        return image[:, bbox.y_min : bbox.y_max + 1, bbox.x_min : bbox.x_max + 1]

    def pad_to_square(self, image: ImageNumpy, *, padding_value: int = 0) -> ImageNumpy:
        """Pad the image to ensure it is a square."""
        # If the image is already square, return it
        if image.shape[1] == image.shape[2]:
            return image

        # Calculate the padding that needs to go before and after the image
        difference = abs(image.shape[1] - image.shape[2])
        padding_before = difference // 2
        padding_after = difference - padding_before

        # Create the padding width
        if image.shape[1] > image.shape[2]:
            padding = ((0, 0), (0, 0), (padding_before, padding_after))
        else:
            padding = ((0, 0), (padding_before, padding_after), (0, 0))

        # Pad the image
        padded_image = np.pad(image, padding, mode="constant", constant_values=padding_value)

        # Make sure the padded image is square
        if padded_image.shape[1] != padded_image.shape[2]:
            raise ValueError(
                "The padded image is not square! Something has gone wrong in the padding process"
                " and there is likely a bug."
            )

        return padded_image
