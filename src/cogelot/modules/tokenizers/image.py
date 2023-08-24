import cv2
import numpy as np
import torch
from einops import rearrange

from cogelot.structures.common import Bbox, ImageType, Observation, View
from cogelot.structures.token import ImageToken, VisualObject


DEFAULT_IMAGE_SIZE = 32
MIN_DIMENSION_SIZE = 2


def pad_image_to_square(image: torch.Tensor, *, padding_value: int = 0) -> torch.Tensor:
    """Pad the image to ensure it is a square."""
    # If the image is already square, return it
    if image.shape[1] == image.shape[2]:
        return image

    dimension_to_pad = 1 if image.shape[1] < image.shape[2] else 0

    # Calculate the padding that needs to go before and after the image
    difference = abs(image.shape[1] - image.shape[2])
    padding_before = difference // 2
    padding_after = difference - padding_before

    # Specify the padding for each dimension
    padding: list[tuple[int, int]] = [(0, 0), (0, 0), (0, 0)]
    padding[dimension_to_pad] = (padding_before, padding_after)

    padded_image = torch.nn.functional.pad(
        image,
        tuple(torch.tensor(padding).flatten().tolist()),
        mode="constant",
        value=padding_value,
    )

    return padded_image


def crop_to_bounding_box(image: torch.Tensor, bbox: Bbox) -> torch.Tensor:
    """Crop the image to the bounding box."""
    return image[:, bbox.y_min : bbox.y_max + 1, bbox.x_min : bbox.x_max + 1]  # noqa: WPS221


def resize_image(image: torch.Tensor, *, image_size: int = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    """Resize the image."""
    image = rearrange(image, "c h w -> h w c")
    image = image.to(torch.float32)
    resized_image = cv2.resize(
        src=image.numpy(),  # pyright: ignore[reportGeneralTypeIssues]
        dsize=(image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )
    image = torch.from_numpy(resized_image)
    image = rearrange(image, "h w c -> c h w")
    return image


def extract_object_from_image(
    image: torch.Tensor, bbox: Bbox, *, image_size: int = DEFAULT_IMAGE_SIZE
) -> torch.Tensor:
    """Crop the object from the RGB image using the bounding box."""
    cropped_image = crop_to_bounding_box(image, bbox)
    # Make sure the cropped image is square
    cropped_image = pad_image_to_square(cropped_image)
    # Resize the image
    cropped_image = resize_image(cropped_image, image_size=image_size)
    return cropped_image


def create_bounding_box_from_segmentation_image(
    object_id: int,
    segmentation_image: torch.Tensor,
    *,
    min_dimension_size: int = MIN_DIMENSION_SIZE,
) -> Bbox:
    """Extract the bounding box for the object ID from the segmentation view."""
    # Get all the pixels that belong to the object
    ys, xs = np.nonzero(segmentation_image == object_id)

    # Make sure that the object is not too small to be cropped
    if len(xs) < min_dimension_size or len(ys) < min_dimension_size:
        raise ValueError("The object is too small to be cropped!")

    # Create the bounding box
    bbox = Bbox.from_abs_xyxy(
        x_min=int(np.min(xs)),
        x_max=int(np.max(xs)),
        y_min=int(np.min(ys)),
        y_max=int(np.max(ys)),
    )
    return bbox


def extract_objects_from_images(
    *,
    image_per_type_per_view: dict[View, dict[ImageType, torch.Tensor]],
    available_object_ids: set[int],
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> list[VisualObject]:
    """Extract objects from images and create a visual token for the given view."""
    visual_objects: list[VisualObject] = []

    for view, images_per_type in image_per_type_per_view.items():
        for object_id in available_object_ids:
            # Extract the bounding box for the object ID from the segmentation image
            try:
                bbox = create_bounding_box_from_segmentation_image(
                    object_id, images_per_type[ImageType.segmentation]
                )
            except ValueError:
                # If the object is too small, then we skip it
                continue

            # Crop the object from the RGB image using the bounding box
            cropped_image = extract_object_from_image(
                images_per_type[ImageType.rgb], bbox, image_size=image_size
            )

            # Create the visual object
            visual_objects.append(
                VisualObject(view=view, bbox=bbox, cropped_image=torch.from_numpy(cropped_image))
            )

    return visual_objects


def create_image_token_from_objects(
    *,
    token_position_idx: int,
    objects: list[VisualObject],
    token_value: str | None = None,
) -> ImageToken:
    """Create a visual token."""
    return ImageToken(token=token_value, index=token_position_idx, objects=objects)


def create_image_token_from_images(
    *,
    token_position_idx: int,
    image_per_type_per_view: dict[View, dict[ImageType, torch.Tensor]],
    available_object_ids: set[int],
    token_value: str | None = None,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> ImageToken:
    """Create a visual token from the raw images."""
    all_visual_objects = extract_objects_from_images(
        image_per_type_per_view=image_per_type_per_view,
        available_object_ids=available_object_ids,
        image_size=image_size,
    )
    visual_token = create_image_token_from_objects(
        token_position_idx=token_position_idx,
        token_value=token_value,
        objects=all_visual_objects,
    )
    return visual_token


class ImageTokenizer:
    """Create image tokens."""

    def __init__(self, image_size: int = DEFAULT_IMAGE_SIZE) -> None:
        self.image_size = image_size

        self.tokenize_images = create_image_token_from_images

    def tokenize_observation(
        self,
        *,
        observation: Observation,
        all_object_ids: set[int],
        token_value: str | None = None,
    ) -> ImageToken:
        """Tokenize an observation."""
        return create_image_token_from_images(
            token_position_idx=observation.index,
            token_value=token_value,
            image_per_type_per_view=observation.to_image_per_type_per_view(),
            available_object_ids=all_object_ids,
        )
