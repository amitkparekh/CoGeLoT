import itertools
from typing import Protocol

import torch
from einops import rearrange, repeat


def stitch_observations_with_actions_using_loops(
    encoded_observations: torch.Tensor,
    encoded_observations_mask: torch.Tensor,
    encoded_actions: torch.Tensor | None,
    encoded_actions_mask: torch.Tensor | None,
    *,
    num_action_tokens_per_timestep: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stitch the observations together with actions using zips and loops.

    This is a lot slower but incredible useful to make sure that the other, more faster, methods
    work as intended.
    """
    _, observation_seq_len, max_objects, _ = encoded_observations.shape
    actions_seq_len = 0 if encoded_actions is None else encoded_actions.shape[1]
    total_seq_len = (
        observation_seq_len * max_objects + actions_seq_len * num_action_tokens_per_timestep
    )

    if observation_seq_len not in {actions_seq_len, actions_seq_len + 1}:
        raise AssertionError(
            "The number of observations must be equal to or one more than the number of actions"
        )

    observation_per_timestep = encoded_observations.split(1, dim=1)
    observation_mask_per_timestep = encoded_observations_mask.split(1, dim=1)

    actions_per_timestep = encoded_actions.split(1, dim=1) if encoded_actions is not None else []
    actions_mask_per_timestep = (
        encoded_actions_mask.split(1, dim=1) if encoded_actions_mask is not None else []
    )

    token_tensors_list = list(
        itertools.chain.from_iterable(
            itertools.zip_longest(observation_per_timestep, actions_per_timestep)
        )
    )
    masks_tensors_list = list(
        itertools.chain.from_iterable(
            itertools.zip_longest(observation_mask_per_timestep, actions_mask_per_timestep)
        )
    )

    token_tensors_list = [tensor for tensor in token_tensors_list if tensor is not None]
    masks_tensors_list = [tensor for tensor in masks_tensors_list if tensor is not None]

    tokens = torch.cat(token_tensors_list, dim=2).squeeze(1)
    masks = torch.cat(masks_tensors_list, dim=2).squeeze(1)
    assert tokens.size(1) == total_seq_len
    assert masks.size(1) == total_seq_len

    return tokens, masks


def create_indices_for_scatter(
    num_timesteps: int,
    num_tokens_for_encoding_type: int,
    index_offset_for_encoding_type: int,
    num_tokens_per_timestep: int,
    device: torch.device,
) -> torch.Tensor:
    """Create indices that are used for the scatter operation when stitching.

    This basically creates a bunch of index numbers which are used to allocate each number from the
    timestep to the stitched tensor.

    Using the scatter operation is better because it removes the for-loop, which can easily add
    time.
    """
    # Create an index from 0 to the number of tokens for every timestep
    token_index = torch.arange(num_tokens_for_encoding_type * num_timesteps, device=device).view(
        -1, num_tokens_for_encoding_type
    )
    # Assign the timestep index to each and every grouping of tokens
    timestep_index = (
        torch.arange(num_timesteps, device=device).repeat(num_tokens_for_encoding_type, 1).T
    )
    # Each token per timestep needs offsetting by a constant so they're in the right position
    timestep_offset_index = timestep_index * (
        num_tokens_per_timestep - num_tokens_for_encoding_type
    )
    # When we add them together, we get the new index position for each token timestep observed
    new_index = token_index + timestep_offset_index + index_offset_for_encoding_type
    # And then flatten it to 1D
    return new_index.flatten()


class AddEncodingToTokensFn(Protocol):
    """Protocol for the fn to stitch together the encodings to the tokens.

    This is so that the observations and actions can be interleaved efficiently.
    """

    def __call__(
        self,
        *,
        encoding: torch.Tensor,
        encoding_mask: torch.Tensor,
        num_tokens_per_timestep: int,
        index_offset_for_encoding_type: int,
        tokens: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add the encoded information to the token placeholders.

        The `num_tokens_per_timestep` are the total number of tokens across all the various
        information types.
        """
        ...


def add_encoding_to_tokens_using_subslice_loop(
    *,
    encoding: torch.Tensor,
    encoding_mask: torch.Tensor,
    num_tokens_per_timestep: int,
    index_offset_for_encoding_type: int,
    tokens: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add the encoded information and masks into the right positions using for-loop.

    This is slower but is kept for testing.
    """
    # Get the information from the shapes
    num_tokens_for_encoding_type = encoding.size(2)

    # Rearrange the tensors to be in the right structure
    encoding = rearrange(encoding, "B L Q E -> (L Q) B E")
    encoding_mask = rearrange(encoding_mask, "B L Q -> (L Q) B")

    for token_idx in range(num_tokens_for_encoding_type):
        output_start_index = token_idx + index_offset_for_encoding_type
        tokens[output_start_index::num_tokens_per_timestep] = encoding[
            token_idx::num_tokens_for_encoding_type
        ]
        masks[output_start_index::num_tokens_per_timestep] = encoding_mask[
            token_idx::num_tokens_for_encoding_type
        ]

    return tokens, masks


def add_encoding_to_tokens_using_scatter(
    *,
    encoding: torch.Tensor,
    encoding_mask: torch.Tensor,
    num_tokens_per_timestep: int,
    index_offset_for_encoding_type: int,
    tokens: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill in the encoded information and masks into the correct positions.

    Use `torch.scatter` to fill them in so that we don't need to use subscript slicing with a for-loop.
    """
    # Get the information from the shapes
    num_timesteps, num_tokens_for_encoding_type = encoding.shape[1:3]

    # Rearrange the tensors to be in the right structure
    encoding = rearrange(encoding, "B L Q E -> (L Q) B E")
    encoding_mask = rearrange(encoding_mask, "B L Q -> (L Q) B")

    new_token_index_for_scatter = create_indices_for_scatter(
        num_timesteps,
        num_tokens_for_encoding_type,
        index_offset_for_encoding_type,
        num_tokens_per_timestep,
        device=encoding.device,
    )

    encoding_scatter_indices = repeat(
        new_token_index_for_scatter, "i -> i bsz dim", bsz=encoding.size(1), dim=encoding.size(2)
    )
    mask_scatter_indices = repeat(
        new_token_index_for_scatter, "i -> i bsz", bsz=encoding_mask.size(1)
    )

    tokens = torch.scatter(tokens, 0, encoding_scatter_indices, encoding)
    masks = torch.scatter(masks, 0, mask_scatter_indices, encoding_mask)
    return tokens, masks


def stitch_observations_with_actions(
    encoded_observations: torch.Tensor,
    encoded_observations_mask: torch.Tensor,
    encoded_actions: torch.Tensor | None,
    encoded_actions_mask: torch.Tensor | None,
    *,
    add_encoding_to_tokens_fn: AddEncodingToTokensFn,
    num_action_tokens_per_timestep: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stitch the observations together with actions for decoder input."""
    batch_size, observation_seq_len, max_objects, embed_dim = encoded_observations.shape
    actions_seq_len = 0 if encoded_actions is None else encoded_actions.shape[1]
    total_seq_len = (
        observation_seq_len * max_objects + actions_seq_len * num_action_tokens_per_timestep
    )

    if observation_seq_len not in {actions_seq_len, actions_seq_len + 1}:
        raise AssertionError(
            "The number of observations must be equal to or one more than the number of actions"
        )

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

    # Add the observations to the tokens and the masks
    tokens, masks = add_encoding_to_tokens_fn(
        encoding=encoded_observations,
        encoding_mask=encoded_observations_mask,
        num_tokens_per_timestep=max_objects + num_action_tokens_per_timestep,
        index_offset_for_encoding_type=0,
        tokens=tokens,
        masks=masks,
    )

    # If we have actions, then we add them in too
    if encoded_actions is not None and encoded_actions_mask is not None:
        tokens, masks = add_encoding_to_tokens_fn(
            encoding=encoded_actions,
            encoding_mask=encoded_actions_mask,
            num_tokens_per_timestep=max_objects + num_action_tokens_per_timestep,
            index_offset_for_encoding_type=max_objects,
            tokens=tokens,
            masks=masks,
        )

    # Put the batch first
    tokens = rearrange(tokens, "L B E -> B L E")
    masks = rearrange(masks, "L B -> B L")

    return tokens, masks
