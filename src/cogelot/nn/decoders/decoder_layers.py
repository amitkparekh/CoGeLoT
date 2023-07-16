from collections.abc import Callable

import torch

from cogelot.nn.decoders.interfaces import TransformerDecoderLayerProtocol


class GeGLUTransformerDecoderLayer(
    torch.nn.TransformerDecoderLayer, TransformerDecoderLayerProtocol
):
    """Transformer Decoder Layer with GeGLU activation."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[torch.Tensor], torch.Tensor] = torch.nn.functional.gelu,
        layer_norm_eps: float = 0.00001,
        batch_first: bool = False,  # noqa: FBT001, FBT002
        norm_first: bool = False,  # noqa: FBT001, FBT002
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            device,
            dtype,
        )

        # Bias is disabled to be identical to their for this part, but it should be enabled
        self.geglu_gated_layer = torch.nn.Linear(
            d_model, dim_feedforward, bias=False, device=device, dtype=dtype
        )

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward block with GeGLU activation function."""
        x = self.linear2(
            self.dropout(self.activation(self.linear1(x)) * self.geglu_gated_layer(x))
        )
        return self.dropout3(x)


class XAttentionOnlyGeGLUTransformerDecoderLayer(
    GeGLUTransformerDecoderLayer, TransformerDecoderLayerProtocol
):
    """Transformer Decoder Layer with GeGLU activation and only using cross-attention.

    In this case, the self-attention block is replaced by a zero tensor.
    """

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,  # noqa: ARG002
        key_padding_mask: torch.Tensor | None,  # noqa: ARG002
        is_causal: bool = False,  # noqa: ARG002, FBT001, FBT002
    ) -> torch.Tensor:
        return torch.zeros_like(x)


class SelfAttnOnlyGeGLUTransformerDecoderLayer(
    GeGLUTransformerDecoderLayer, TransformerDecoderLayerProtocol
):
    """Transformer Decoder Layer with GeGLU activation and only using self-attention.

    In this case, the cross-attention block is replaced by a zero tensor.
    """

    def _mha_block(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,  # noqa: ARG002
        attn_mask: torch.Tensor | None,  # noqa: ARG002
        key_padding_mask: torch.Tensor | None,  # noqa: ARG002
        is_causal: bool = False,  # noqa: FBT002, FBT001, ARG002
    ) -> torch.Tensor:
        return torch.zeros_like(x)


class VIMATransformerDecoderLayer(TransformerDecoderLayerProtocol):
    """Alternative VIMA Transformer Decoder Layer, but in torch.

    This does the same iterating over the dedicated xattn and sattn layers separately.
    """

    def __init__(
        self,
        embed_dim: int,
        sattn_n_heads: int = 12,
        xattn_n_heads: int = 8,
        dim_feedforward_multiplier: int = 4,
        dropout: float = 0.1,
        *,
        batch_first: bool = True,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.xattn_layer = XAttentionOnlyGeGLUTransformerDecoderLayer(
            d_model=embed_dim,
            nhead=xattn_n_heads,
            dim_feedforward=embed_dim * dim_feedforward_multiplier,
            dropout=dropout,
            batch_first=batch_first,
            norm_first=norm_first,
        )

        self.sattn_layer = SelfAttnOnlyGeGLUTransformerDecoderLayer(
            d_model=embed_dim,
            nhead=sattn_n_heads,
            dim_feedforward=embed_dim * dim_feedforward_multiplier,
            dropout=dropout,
            batch_first=batch_first,
            norm_first=norm_first,
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        *,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:
        """Forward method, calling both the xattn and the sattn."""
        x = tgt
        x = self.xattn_layer(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=tgt_is_causal,
            memory_is_causal=memory_is_causal,
        )
        x = self.sattn_layer(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=tgt_is_causal,
            memory_is_causal=memory_is_causal,
        )
        return x
