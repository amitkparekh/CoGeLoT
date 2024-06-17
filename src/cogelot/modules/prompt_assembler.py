from collections.abc import Iterator
from typing import Literal, NamedTuple, cast

import torch

from cogelot.structures.model import RawPromptTokenType
from vima.utils import DataDict

EmbeddingWithPosition = tuple[torch.Tensor, torch.Tensor]
VisualsMask = dict[int, dict[Literal["front", "top"], torch.Tensor]]


class AssembledModality(NamedTuple):
    """A modality that has been assembled."""

    embedding: list[EmbeddingWithPosition]
    mask: list[EmbeddingWithPosition]


def _assemble_tensor(
    text: list[EmbeddingWithPosition], visuals: list[EmbeddingWithPosition]
) -> torch.Tensor:
    """Assemble the multimodal prompt.

    Merge the two lists together, and sort them by position, and just get the embeddings (removing
    the index)
    """
    return torch.cat([element[1] for element in sorted([*text, *visuals])], dim=0)


def assemble_text(
    *,
    raw_prompts_token_type: RawPromptTokenType,
    embedded_text: torch.Tensor,
    mask: torch.Tensor | None,
    device: torch.device,
) -> Iterator[AssembledModality]:
    """Assemble the text prompt for the given batch."""
    if mask is None:
        mask = torch.ones(embedded_text.shape[:-1], dtype=torch.bool, device=device)

    for batch_idx, raw_prompt_tokens in enumerate(raw_prompts_token_type):
        word_positions = torch.tensor(raw_prompt_tokens, device=device).eq(0).nonzero().flatten()
        word_embedded_per_position = embedded_text[batch_idx, : word_positions.size(0)]
        word_embedding_with_positions: list[EmbeddingWithPosition] = list(
            zip(word_positions, word_embedded_per_position.split(1), strict=True)
        )

        word_mask_per_position = mask[batch_idx, : word_positions.size(0)]
        word_masks_with_positions: list[EmbeddingWithPosition] = list(
            zip(word_positions, word_mask_per_position.split(1), strict=True)
        )
        yield AssembledModality(
            embedding=word_embedding_with_positions, mask=word_masks_with_positions
        )


def assemble_visuals(
    *,
    embedded_visuals: torch.Tensor,
    visuals_mask: dict[int, dict[Literal["front", "top"], torch.Tensor]] | None,
    raw_prompts_token_type: RawPromptTokenType,
    embed_dim: int,
    device: torch.device,
) -> Iterator[AssembledModality]:
    """Assemble the visuals for the multimodal prompt."""
    for batch_idx, raw_prompt_tokens in enumerate(raw_prompts_token_type):
        image_positions = torch.tensor(raw_prompt_tokens, device=device).eq(1).nonzero().flatten()
        num_images = len(image_positions)

        embedded_images = embedded_visuals[batch_idx, :num_images]
        # If there is no visuals_mask, it means that we are using patches so just create a mask
        # that assumes all tokens are valid.
        mask_per_image = (
            (
                torch.cat(
                    [element[1] for element in sorted(visuals_mask[batch_idx].items())],
                    dim=-1,
                )[:num_images]
                .flatten()
                .chunk(num_images)
            )
            if visuals_mask is not None
            else torch.ones(num_images, embedded_images.size(1), dtype=torch.bool, device=device)
            .flatten()
            .chunk(num_images)
        )

        image_embedding_with_positions: list[EmbeddingWithPosition] = list(
            zip(
                image_positions,
                embedded_images.view(-1, embed_dim).chunk(num_images),
                strict=True,
            )
        )

        image_mask_with_positions: list[EmbeddingWithPosition] = list(
            zip(image_positions, mask_per_image, strict=True)
        )
        yield AssembledModality(
            embedding=image_embedding_with_positions, mask=image_mask_with_positions
        )


def assemble_multimodal_prompt(
    *,
    embedded_text: torch.Tensor,
    text_mask: torch.Tensor | None,
    embedded_visuals: torch.Tensor,
    original_visuals: DataDict,
    raw_prompts_token_type: RawPromptTokenType,
    embed_dim: int,
    device: torch.device,
    is_using_patches: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble the multimodal prompt, interleaving the text and visuals."""
    text_iterator = assemble_text(
        raw_prompts_token_type=raw_prompts_token_type,
        embedded_text=embedded_text,
        mask=text_mask,
        device=device,
    )
    visuals_iterator = assemble_visuals(
        embedded_visuals=embedded_visuals,
        # TODO: If using patches, we should not be using the visuals_mask at all. this needs fixing.
        visuals_mask=None if is_using_patches else cast(VisualsMask, original_visuals["mask"]),
        raw_prompts_token_type=raw_prompts_token_type,
        embed_dim=embed_dim,
        device=device,
    )

    prompt_tokens: list[torch.Tensor] = []
    prompt_masks: list[torch.Tensor] = []

    for assembled_text, assembled_visuals in zip(text_iterator, visuals_iterator, strict=True):
        prompt_tokens.append(
            _assemble_tensor(assembled_text.embedding, assembled_visuals.embedding)
        )
        prompt_masks.append(_assemble_tensor(assembled_text.mask, assembled_visuals.mask))

    prompt_tokens_tensor = torch.nn.utils.rnn.pad_sequence(prompt_tokens, batch_first=True)
    prompt_masks_tensor = torch.nn.utils.rnn.pad_sequence(prompt_masks, batch_first=True)

    # Convert to the PyTorch-style mask, where True means it IS MASKED. The VIMA source opts
    # for the other approach, and we are going to be consistent dammit.
    prompt_masks_tensor = ~prompt_masks_tensor
    return prompt_tokens_tensor, prompt_masks_tensor
