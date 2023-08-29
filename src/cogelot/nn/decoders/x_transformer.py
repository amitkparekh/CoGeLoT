from typing import Any

import torch
from x_transformers import ContinuousTransformerWrapper

from cogelot.nn.decoders.interfaces import TransformerDecoderProtocol
from cogelot.nn.embedder import EmbedderProtocol


def _create_padding_mask_from_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Create a padding mask from a tensor."""
    return torch.ones(
        size=tensor.shape[:-1],
        dtype=torch.bool,
        device=tensor.device,
    )


class TransformerDecoder(TransformerDecoderProtocol, ContinuousTransformerWrapper):  # type: ignore[misc]
    """XTransformers wrapper for the decoder.

    This is a wrapper around the `ContinuousTransformerWrapper` from the `x-transformers` library,
    so that the signature of the forward is the same.
    """

    def __init__(self, xattn_embedder: EmbedderProtocol, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.xattn_embedder = xattn_embedder

    def _add_position_embedding_to_memory(
        self, memory: torch.Tensor, memory_key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Add the position embedding to the memory, a.k.a.

        the cross-attention input.
        """
        memory_position_ids = (torch.cumsum(memory_key_padding_mask, dim=1) - 1).long()
        embedded_memory_position = self.xattn_embedder.forward(memory, memory_position_ids)
        memory_with_position = memory + embedded_memory_position
        return memory_with_position


class PromptEncoderHistoryDecoder(TransformerDecoder):
    """XTransformers wrapper for the encoder-decoder, with the prompt in the encoder."""

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,  # noqa: ARG002
        memory_mask: torch.Tensor | None = None,  # noqa: ARG002
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the decoder."""
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = _create_padding_mask_from_tensor(tgt)
        if memory_key_padding_mask is None:
            memory_key_padding_mask = _create_padding_mask_from_tensor(memory)

        # Create the position ids from the mask (just how they do in the Policy)
        position_ids = (torch.cumsum(tgt_key_padding_mask, dim=1) - 1).long()

        memory = self._add_position_embedding_to_memory(memory, memory_key_padding_mask)

        model_output = ContinuousTransformerWrapper.forward(
            self,
            x=tgt,
            context=memory,
            mask=tgt_key_padding_mask,
            context_mask=memory_key_padding_mask,
            pos=position_ids,
        )
        assert isinstance(model_output, torch.Tensor)
        return model_output


class ConventionalEncoderDecoder(TransformerDecoder):
    """XTransformers wrapper for an encoder-decoder model, but 'normally'.

    This is the more conventional way of doing it: by putting the prompt and all the history as
    input from cross-attention, and just decoding the outputs.
    """

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the decoder."""
        raise NotImplementedError


class DecoderOnly(TransformerDecoder):
    """XTransformers wrapper for the decoder-only model.

    For any given input, the memory is prepended to the target sequence so that it all goes in the
    decoder together.

    This is a wrapper around the `ContinuousTransformerWrapper` from the `x-transformers` library,
    so that the signature of the forward is the same.
    """

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,  # noqa: ARG002
        memory_mask: torch.Tensor | None = None,  # noqa: ARG002
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the decoder."""
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = _create_padding_mask_from_tensor(tgt)
        if memory_key_padding_mask is None:
            memory_key_padding_mask = _create_padding_mask_from_tensor(memory)

        # Add separate position embeddings to the memory
        memory = self._add_position_embedding_to_memory(memory, memory_key_padding_mask)

        # Prepend the memory to the target sequence
        tgt = torch.cat([memory, tgt], dim=1)
        tgt_key_padding_mask = torch.cat([memory_key_padding_mask, tgt_key_padding_mask], dim=1)

        # Create the position ids from the mask (just how they do in the Policy)
        position_ids = (torch.cumsum(tgt_key_padding_mask, dim=1) - 1).long()

        model_output = ContinuousTransformerWrapper.forward(
            self,
            x=tgt,
            mask=tgt_key_padding_mask,
            pos=position_ids,
        )
        assert isinstance(model_output, torch.Tensor)
        return model_output
