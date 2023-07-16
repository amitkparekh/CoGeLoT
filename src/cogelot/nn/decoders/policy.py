import torch

from cogelot.nn.decoders.interfaces import (
    TransformerDecoderLayerProtocol,
    TransformerDecoderProtocol,
)


# TODO: improve the name. This is a bit confusing.
class PolicyTransformerDecoder(TransformerDecoderProtocol):
    """Transformer Decoder for the policy using torch."""

    def __init__(
        self,
        *,
        transformer_decoder_layer: TransformerDecoderLayerProtocol,
        embed_dim: int = 768,
        model_seq_length: int = 512,
        n_layers: int = 12,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self._transformer_decoder = torch.nn.TransformerDecoder(
            decoder_layer=transformer_decoder_layer, num_layers=n_layers
        )

        self._memory_abs_pos_embed = torch.nn.Embedding(model_seq_length, embed_dim)
        self._tgt_abs_pos_embed = torch.nn.Embedding(model_seq_length, embed_dim)

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
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = torch.ones(
                size=tgt.shape[:-1],
                dtype=torch.bool,
                device=tgt.device,
            )
        if memory_key_padding_mask is None:
            memory_key_padding_mask = torch.ones(
                size=memory.shape[:-1],
                dtype=torch.bool,
                device=memory.device,
            )

        # Create position embeddings for the memory input (xattn input)
        memory_position_ids = memory_key_padding_mask.cumsum(dim=-1) - 1
        embed_memory_position: torch.Tensor = self._memory_abs_pos_embed(memory_position_ids)

        # Create position embeddings for the target input (tgt input)
        target_position_ids = tgt_key_padding_mask.cumsum(dim=-1) - 1
        embed_target_position: torch.Tensor = self._tgt_abs_pos_embed(target_position_ids)

        # add the pos embedding onto the tensors
        embed_memory = memory + embed_memory_position
        embed_target = tgt + embed_target_position

        # pass the memory and target through the transformer decoder
        transformer_output: torch.Tensor = self._transformer_decoder(
            tgt=embed_target,
            memory=embed_memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        return transformer_output
