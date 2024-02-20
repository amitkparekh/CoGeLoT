import torch
from torch.nn.modules.transformer import _generate_square_subsequent_mask  # noqa: WPS450

from cogelot.nn.decoders.interfaces import TransformerDecoderProtocol
from cogelot.nn.decoders.x_transformer import create_padding_mask_from_tensor


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

        if tgt_mask is None and self._use_casual_mask:
            tgt_mask = _generate_square_subsequent_mask(tgt.size(1), device=tgt.device)

        transformer_output = self.decoder(
            tgt=tgt_with_position,
            memory=memory_with_position,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=True,
        )
        assert isinstance(transformer_output, torch.Tensor)
        return transformer_output
