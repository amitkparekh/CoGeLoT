import torch
from loguru import logger
from torch.nn.modules.transformer import _generate_square_subsequent_mask  # noqa: WPS450

from cogelot.nn.decoders.interfaces import TransformerDecoderProtocol


def create_padding_mask_from_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Create a padding mask from a tensor."""
    return torch.ones(
        size=tensor.shape[:-1],
        dtype=torch.bool,
        device=tensor.device,
    )


def _convert_to_float_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert a boolean mask to a float mask."""
    return mask.to(dtype).masked_fill(mask.logical_not(), torch.finfo(dtype).min)


class TorchVanillaDecoder(TransformerDecoderProtocol):
    """Transformer decoder with torch."""

    def __init__(
        self,
        *,
        decoder: torch.nn.TransformerDecoder,
        pos_embedder: torch.nn.Embedding,
        xattn_embedder: torch.nn.Embedding,
        use_casual_mask: bool = True,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.pos_embedder = pos_embedder
        self.xattn_embedder = xattn_embedder
        self._use_casual_mask = use_casual_mask

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass."""
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = create_padding_mask_from_tensor(tgt)
        if memory_key_padding_mask is None:
            memory_key_padding_mask = create_padding_mask_from_tensor(memory)
        if tgt_mask is None and self._use_casual_mask:
            tgt_mask = _generate_square_subsequent_mask(tgt.size(1), device=tgt.device)

        # Create the position ids from the mask (just how they do in the Policy)
        position_ids = torch.cumsum(tgt_key_padding_mask, dim=1)
        position_ids = position_ids.long()
        tgt_position_embedding = self.pos_embedder(position_ids)
        tgt_with_position = tgt + tgt_position_embedding

        memory_position_ids = (
            (torch.cumsum(~memory_key_padding_mask, dim=1) - 1).long().clamp(min=0)
        )
        embedded_memory_position = self.xattn_embedder(memory_position_ids)
        memory_with_position = memory + embedded_memory_position

        # Make things float
        # Note for future me when I panic: If the mask is TRUE, then it means it _should_ be
        # masked and set to -inf. If it's false, then it's should turn into 0
        tgt_key_padding_mask = _convert_to_float_mask(
            tgt_key_padding_mask, tgt_with_position.dtype
        )
        memory_key_padding_mask = _convert_to_float_mask(
            memory_key_padding_mask, memory_with_position.dtype
        )

        transformer_output = self.decoder(
            tgt=tgt_with_position,
            memory=memory_with_position,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=self._use_casual_mask,
        )
        assert isinstance(transformer_output, torch.Tensor)
        return transformer_output


class TorchDecoderOnly(TransformerDecoderProtocol):
    """Decoder-only transformer (no cross-attention)."""

    def __init__(
        self,
        *,
        encoder: torch.nn.TransformerEncoder,
        pos_embedder: torch.nn.Embedding,
        use_causal_mask: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pos_embedder = pos_embedder
        self._use_causal_mask = use_causal_mask

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass."""
        if tgt_mask is not None or memory_mask is not None:
            logger.warning("Ignoring the provided masks for the decoder-only model.")

        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = create_padding_mask_from_tensor(tgt)
        if memory_key_padding_mask is None:
            memory_key_padding_mask = create_padding_mask_from_tensor(memory)

        decoder_input = torch.cat([memory, tgt], dim=1)
        decoder_key_padding_mask = torch.cat(
            [memory_key_padding_mask, tgt_key_padding_mask], dim=1
        )
        causal_mask = _generate_square_subsequent_mask(
            decoder_input.size(1), device=decoder_input.device
        )

        # Create the position ids from the mask (just how they do in the Policy)
        position_ids = torch.cumsum(decoder_key_padding_mask, dim=1)
        position_ids = position_ids.long()
        input_position_embedding = self.pos_embedder(position_ids)
        input_with_position = decoder_input + input_position_embedding

        transformer_output = self.encoder(
            src=input_with_position,
            src_key_padding_mask=decoder_key_padding_mask,
            mask=causal_mask,
        )
        assert isinstance(transformer_output, torch.Tensor)
        return transformer_output
