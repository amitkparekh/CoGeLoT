from typing import cast

import torch

from cogelot.data.collate import ImageFeatures
from vima.utils import DataDict, any_to_datadict


def _apply_reordering_to_feature(
    feature_tensor: torch.Tensor, mask: torch.Tensor, new_object_order: torch.Tensor
) -> torch.Tensor:
    shuffled_feature = feature_tensor.masked_scatter(
        mask.unsqueeze(-1), feature_tensor[mask][new_object_order]
    )
    return shuffled_feature


@torch.no_grad
def compute_new_object_order(mask: torch.Tensor) -> torch.Tensor:
    """Compute the new order of objects that we can use with a masked scatter."""
    old_object_order = torch.arange(1, mask.sum().item() + 1, device=mask.device)
    old_order_in_batch = torch.zeros_like(mask, dtype=torch.long).masked_scatter_(
        mask, old_object_order
    )
    # Shuffle the order across the object dimension, which will make the order different for
    # each observation within a batch.
    shuffle_order_across_obj_dim = old_order_in_batch.take_along_dim(
        torch.rand(*old_order_in_batch.shape, device=old_order_in_batch.device).sort()[1],
        dim=-1,
    )
    new_object_order = shuffle_order_across_obj_dim[shuffle_order_across_obj_dim != 0]
    return new_object_order - 1


def shuffle_objects_for_each_observation(objects: ImageFeatures | DataDict) -> DataDict:
    """Shuffle the object tokens for each observation."""
    objects = cast(ImageFeatures, objects)

    # Since the mask is the same for both the front and top views, it doesn't matter
    mask = objects["mask"]["front"]
    new_object_order = compute_new_object_order(mask)

    img_front = objects["cropped_img"]["front"]
    img_top = objects["cropped_img"]["top"]
    bbox_front = objects["bbox"]["front"]
    bbox_top = objects["bbox"]["top"]

    transformed_observation = objects
    transformed_observation["cropped_img"]["front"] = _apply_reordering_to_feature(
        img_front, mask, new_object_order
    )
    transformed_observation["cropped_img"]["top"] = _apply_reordering_to_feature(
        img_top, mask, new_object_order
    )
    transformed_observation["bbox"]["front"] = _apply_reordering_to_feature(
        bbox_front, mask, new_object_order
    )
    transformed_observation["bbox"]["top"] = _apply_reordering_to_feature(
        bbox_top, mask, new_object_order
    )
    return any_to_datadict(transformed_observation)
