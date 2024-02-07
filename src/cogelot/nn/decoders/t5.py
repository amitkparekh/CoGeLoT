from typing import Literal

import torch
from transformers.models.roformer.modeling_roformer import RoFormerConfig, RoFormerEncoder
from transformers.models.t5.modeling_t5 import T5Config, T5Stack

from cogelot.nn.decoders.interfaces import TransformerDecoderProtocol


class T5Decoder(TransformerDecoderProtocol):
    """T5 but just the decoder."""

    def __init__(
        self,
        embd_dim: int = 768,
        *,
        n_layer: int = 12,
        n_head: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        feed_forward_proj: Literal["relu", "gated-gelu"] = "relu",
    ) -> None:
        super().__init__()

        config = T5Config(
            d_model=embd_dim,
            n_head=n_head,
            num_layers=n_layer,
            dropout_rate=dropout,
            d_ff=d_ff,
            feed_forward_proj=feed_forward_proj,
            is_decoder=True,
            is_encoder_decoder=False,
        )
        self.decoder = T5Stack(config)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,  # noqa: ARG002
        memory_mask: torch.Tensor | None = None,  # noqa: ARG002
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the T5 decoder."""
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = torch.zeros(
                size=tgt.shape[:-1],
                dtype=torch.bool,
                device=tgt.device,
            )
        if memory_key_padding_mask is None:
            memory_key_padding_mask = torch.zeros(
                size=memory.shape[:-1],
                dtype=torch.bool,
                device=memory.device,
            )

        # Since they use HF-style mask meanings, invert the masks
        tgt_key_padding_mask = ~tgt_key_padding_mask
        memory_key_padding_mask = ~memory_key_padding_mask
        transformer_output = self.decoder(
            inputs_embeds=tgt,
            attention_mask=tgt_key_padding_mask,
            encoder_hidden_states=memory,
            encoder_attention_mask=memory_key_padding_mask,
        ).last_hidden_state
        return transformer_output


class RoFormerDecoder(TransformerDecoderProtocol):
    """RoFormer but just the decoder."""

    def __init__(
        self,
        embd_dim: int = 768,
        *,
        n_layer: int = 12,
        n_head: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        hidden_act: Literal["gelu", "gelu_new"] = "gelu_new",
    ) -> None:
        super().__init__()
        config = RoFormerConfig(
            hidden_size=embd_dim,
            num_attention_heads=n_head,
            num_hidden_layers=n_layer,
            dropout_rate=dropout,
            intermediate_size=d_ff,
            is_decoder=True,
            add_cross_attention=True,
            hidden_dropout_prob=dropout,
            hidden_act=hidden_act,
            attention_probs_dropout_prob=dropout,
        )
        self.decoder = RoFormerEncoder(config)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,  # noqa: ARG002
        memory_mask: torch.Tensor | None = None,  # noqa: ARG002
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the T5 decoder."""
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = torch.zeros(
                size=tgt.shape[:-1],
                dtype=torch.bool,
                device=tgt.device,
            )
        if memory_key_padding_mask is None:
            memory_key_padding_mask = torch.zeros(
                size=memory.shape[:-1],
                dtype=torch.bool,
                device=memory.device,
            )

        # Since they use HF-style mask meanings, invert the masks
        tgt_key_padding_mask = ~tgt_key_padding_mask
        memory_key_padding_mask = ~memory_key_padding_mask
        transformer_output = self.decoder(
            hidden_states=tgt,
            attention_mask=tgt_key_padding_mask,
            encoder_hidden_states=memory,
            encoder_attention_mask=memory_key_padding_mask,
            return_dict=False,
        ).last_hidden_state
        return transformer_output
