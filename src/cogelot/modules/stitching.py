from typing import Protocol

import torch
from einops import rearrange


def get_max_num_objects_from_encoded_observations(encoded_observations: torch.Tensor) -> int:
    """Get the maximum number of objects from the encoded observations."""
    return encoded_observations.shape[-2]


def create_indices_for_observation_scatter(
    observation_seq_len: int,
    max_objects_per_observation: int,
    device: torch.device,
) -> torch.Tensor:
    """Create indices that are used for the scatter operation when stitching.

    This basically creates a bunch of index numbers which are used to allocate each number from the
    observation to the stitched tensor.

    Using the scatter operation is better because it removes the for-loop, which can easily add
    time.
    """
    # Create an index from 0 to the total number of objects in the observations
    obj_index = torch.arange(max_objects_per_observation * observation_seq_len, device=device)
    # Assign the observation index to each and every object
    obs_index = (
        torch.arange(observation_seq_len, device=device)
        .repeat(max_objects_per_observation, 1)
        .T.flatten()
    )
    # When we add them together, we get the new index position for each object observed
    new_obj_index = obj_index + obs_index
    return new_obj_index


class AddObservationsToTokensFn(Protocol):
    """Protocol for the fn to add observations to the tokens."""

    def __call__(
        self,
        *,
        encoded_observations: torch.Tensor,
        encoded_observations_mask: torch.Tensor,
        observation_seq_len: int,
        max_objects: int,
        tokens: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add the encoded observations to the token placeholders."""
        ...  # noqa: WPS428


def add_observations_to_tokens_using_loop(
    *,
    encoded_observations: torch.Tensor,
    encoded_observations_mask: torch.Tensor,
    observation_seq_len: int,  # noqa: ARG001
    max_objects: int,
    tokens: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FIll in the encoded observations and masks into the correct position using for-loop.

    This is slower but is kept for testing.
    """
    for obj_idx in range(max_objects):
        tokens[obj_idx :: max_objects + 1] = encoded_observations[  # noqa: WPS362
            obj_idx::max_objects
        ]
        masks[obj_idx :: max_objects + 1] = encoded_observations_mask[  # noqa: WPS362
            obj_idx::max_objects
        ]

    return tokens, masks


def add_observations_to_tokens_using_scatter(
    *,
    encoded_observations: torch.Tensor,
    encoded_observations_mask: torch.Tensor,
    observation_seq_len: int,
    max_objects: int,
    tokens: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill in the encoded observations and masks into the correct positions.

    For the observations, we use `torch.scatter` to fill them in so that we don't need to use
    subscript slicing with a for-loop.
    """
    new_obj_index_for_scatter = create_indices_for_observation_scatter(
        observation_seq_len,
        max_objects,
        device=encoded_observations.device,
    )
    # Shape (obj_seq_len, batch_size, dim)
    observation_scatter_indices = new_obj_index_for_scatter.repeat(
        *encoded_observations.shape[:0:-1], 1
    ).transpose(-1, 0)
    # Shape (obj_seq_len, batch_size)
    mask_scatter_indices = new_obj_index_for_scatter.repeat(
        *encoded_observations_mask.shape[:0:-1], 1
    ).T
    tokens = torch.scatter(tokens, 0, observation_scatter_indices, encoded_observations)
    masks = torch.scatter(masks, 0, mask_scatter_indices, encoded_observations_mask)
    return tokens, masks


def stitch_observations_with_actions(  # noqa: WPS210
    encoded_observations: torch.Tensor,
    encoded_observations_mask: torch.Tensor,
    encoded_actions: torch.Tensor | None,
    encoded_actions_mask: torch.Tensor | None,
    *,
    add_observations_to_tokens_fn: AddObservationsToTokensFn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stitch the observations together with actions for decoder input."""
    embed_dim = encoded_observations.size(-1)
    batch_size, observation_seq_len = encoded_observations.shape[:2]
    actions_seq_len = 0 if encoded_actions is None else encoded_actions.shape[1]

    if observation_seq_len not in {actions_seq_len, actions_seq_len + 1}:
        raise AssertionError(
            "The number of observations must be equal to or one more than the number of actions"
        )

    max_objects = get_max_num_objects_from_encoded_observations(encoded_observations)
    total_seq_len = observation_seq_len * max_objects + actions_seq_len

    # Rearrange the tensors to be in the right structure
    # embedded_observations = rearrange(embedded_observations, "L B Q E -> B L Q E")
    encoded_observations = rearrange(encoded_observations, "B L Q E -> B (L Q) E")
    encoded_observations = rearrange(encoded_observations, "B L E -> L B E")

    # embedded_observations_mask = rearrange(embedded_observations_mask, "L B Q -> B L Q")
    encoded_observations_mask = rearrange(encoded_observations_mask, "B L Q -> B (L Q)")
    encoded_observations_mask = rearrange(encoded_observations_mask, "B L -> L B")

    # Create tensors which will we will use to put the various tokens into
    tokens = torch.empty(
        total_seq_len,
        batch_size,
        embed_dim,
        dtype=torch.float32,
        device=encoded_observations.device,
    )

    # All elements are the mask start off by being masked.
    masks = torch.ones(
        total_seq_len, batch_size, dtype=torch.bool, device=encoded_observations.device
    )

    tokens, masks = add_observations_to_tokens_fn(
        encoded_observations=encoded_observations,
        encoded_observations_mask=encoded_observations_mask,
        observation_seq_len=observation_seq_len,
        max_objects=max_objects,
        tokens=tokens,
        masks=masks,
    )

    if encoded_actions is not None:
        tokens[max_objects :: max_objects + 1] = encoded_actions.transpose(0, 1)  # noqa: WPS362
    if encoded_actions_mask is not None:
        masks[max_objects :: max_objects + 1] = encoded_actions_mask.T  # noqa: WPS362

    # Put the batch first
    tokens = rearrange(tokens, "L B E -> B L E")
    masks = rearrange(masks, "L B -> B L")

    return tokens, masks
