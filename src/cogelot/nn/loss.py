from typing import Self, cast, get_args

import torch
from pydantic import BaseModel
from torch.masked import MaskedTensor, as_masked_tensor

from cogelot.structures.vima import (
    AxesPerPoseActionType,
    PoseActionType,
    PositionAxes,
    RotationAxes,
)
from vima.nn.action_decoder.dists import MultiCategorical


LOSS_KEY_TEMPLATE = "{pose_action_type}_{axis}"


class PerActionPerAxis(BaseModel, arbitrary_types_allowed=True):
    """Dictionary allowing a tensor to be split per axis per action.

    Every single tensor has the following shape: [batch_size, max_timesteps, ...].
    """

    pose0_position: dict[PositionAxes, torch.Tensor]
    pose0_rotation: dict[RotationAxes, torch.Tensor]
    pose1_position: dict[PositionAxes, torch.Tensor]
    pose1_rotation: dict[RotationAxes, torch.Tensor]

    @property
    def batch_size(self) -> int:
        """Get the batch size."""
        return self._get_any_tensor.size(0)

    @property
    def num_timesteps(self) -> int:
        """Get the max num timesteps."""
        return self._get_any_tensor.size(1)

    @classmethod
    def from_actions(cls, actions: dict[PoseActionType, torch.Tensor]) -> Self:
        """Split an action dict per axis and action.

        This will allow for easier loss tracking across the axes and actions, to make sure things
        are moving properly.
        """
        per_action_per_axis = {
            pose_action_type: dict(
                zip(
                    get_args(AxesPerPoseActionType[pose_action_type]),
                    tensor.split(1, dim=-1),
                    strict=True,
                )
            )
            for pose_action_type, tensor in actions.items()
        }
        return cls.parse_obj(per_action_per_axis)

    @classmethod
    def from_action_logits(cls, actions: dict[PoseActionType, list[torch.Tensor]]) -> Self:
        """Split lists of tensors per axis per action into separate tensors.

        This is simpler as we just have to make sure the number of tensors line up with the number
        of axes and then return.
        """
        per_action_per_axis = {
            pose_action_type: dict(
                zip(
                    get_args(AxesPerPoseActionType[pose_action_type]),
                    tensors,
                    strict=True,
                )
            )
            for pose_action_type, tensors in actions.items()
        }
        return cls.parse_obj(per_action_per_axis)

    def to_flattened_dict(self) -> dict[str, torch.Tensor]:
        """Flatten the dict into a single dict."""
        return {
            LOSS_KEY_TEMPLATE.format(pose_action_type=pose_action_type, axis=axis): tensor
            for pose_action_type, tensor_per_axis in self.dict().items()
            for axis, tensor in tensor_per_axis.items()
        }

    @property
    def _get_any_tensor(self) -> torch.Tensor:
        """Get any one of the tensors from the object, doesn't matter which."""
        return next(
            tensor
            for tensor_per_axis in self.dict().values()
            for tensor in tensor_per_axis.values()
        )


def compute_fine_grained_loss(
    predicted_actions: dict[PoseActionType, MultiCategorical],
    target_actions: dict[PoseActionType, torch.Tensor],
    *,
    ignore_target_index: int = -100,
) -> dict[str, MaskedTensor]:
    """Compute a fine-grained loss across all the poses and the axes per pose.

    Since a trajectory can be made of more than one action, we need to average the loss across the
    trajectory _and_ the batch.

    Additionally, the loss for each action is constructed from predictions across each axis for
    each head (e.g. the rotation pose has 4 heads: one for each of X, Y, Z, W), meaning there are
    multiple separate losses to compute and compare.

    On top of calculating the loss, we need a way to ensure the mask from the targets are kept as
    we will need these when reducing the loss. If we don't, the loss will be incorrect. ALthough
    torch's masked tensors are currently in prototype stage, they support autograd and the
    reduction operators what we will be needing, so we _should_ be able to use them without issue.
    """
    predicted_logits_per_pose_per_axis = PerActionPerAxis.from_action_logits(
        cast(
            dict[PoseActionType, list[torch.Tensor]],
            {
                pose_action_type: [axis_dist.logits for axis_dist in action_dist.dists]
                for pose_action_type, action_dist in predicted_actions.items()
            },
        )
    )
    target_per_pose_per_axis = PerActionPerAxis.from_actions(target_actions)

    iterator = zip(
        predicted_logits_per_pose_per_axis.to_flattened_dict().items(),
        target_per_pose_per_axis.to_flattened_dict().items(),
        strict=True,
    )

    loss_per_action_per_axis: dict[str, MaskedTensor] = {}

    # When calculating the loss itself, we need to reshape we to be 2-dimensional (not sure why
    # but it's needed). Afterwards, we can reshape it to the target shape, as we are now in the
    # shape of (batch, num_timesteps, 1). This seems excessive but it's the only way I know how
    # right now because torch docs say that providing class indices to the targets is more
    # performative, and performance is good.
    for (loss_key, predicted_logits), (_, targets) in iterator:
        loss = torch.nn.functional.cross_entropy(
            predicted_logits.reshape(-1, predicted_logits.size(-1)),
            targets.reshape(-1),
            ignore_index=ignore_target_index,
            reduction="none",
        ).reshape(targets.shape)

        # To make sure we mask out the ignored target indices when reducing the loss, we turn the
        # loss tensor into a masked tensor.
        loss = cast(MaskedTensor, as_masked_tensor(loss, mask=targets != ignore_target_index))
        loss_per_action_per_axis[loss_key] = loss

    return loss_per_action_per_axis


def reduce_fine_grained_loss(fine_grained_loss: dict[str, MaskedTensor]) -> torch.Tensor:
    """Reduce the fine-grained loss into a single number for backprop.

    Every single tensor should have the exact same shape: [batch_size, max_timesteps, 1]. This
    means that we should be able to stack them together and just easily manipulate the stack to get
    the overall loss.

    To reduce the loss, we need to sum across the axes for a given timestep, and then we average
    across all the timesteps within a batch, and then across all batches.

    The mean-ing must be performed in two steps since it is not identical to doing it in a single
    step. For example, I ran the following during debugging and they are not identical:

    ```python
    > loss_per_batch_per_timestep_per_axis.sum(dim=-1).mean().item()
    48.338687896728516

    > loss_per_batch_per_timestep_per_axis.sum(dim=-1).mean(dim=-1).mean(dim=-1).item()
    48.33868408203125
    ```
    """
    # Shape: [batch_size, max_timesteps, total_axes_across_actions]
    loss_per_batch_per_timestep_per_axis = cast(
        MaskedTensor, torch.cat(list(fine_grained_loss.values()), dim=-1)
    )

    # Reduce the loss to a single number
    reduced_loss = loss_per_batch_per_timestep_per_axis.sum(dim=-1).mean(dim=-1).mean(dim=-1)
    assert isinstance(reduced_loss, MaskedTensor)
    assert reduced_loss.ndim == 0
    return reduced_loss.get_data()  # pyright: ignore[reportGeneralTypeIssues]
