import abc

import torch


class TransformerDecoderLayerProtocol(abc.ABC, torch.nn.Module):
    """Protocol for Transformer Decoder Layer.

    Ensures any custom layers are identical to torch.nn.TransformerDecoderLayer.
    """

    @abc.abstractmethod
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
        """Forward pass of the decoder layer."""
        ...  # noqa: WPS428


class TransformerDecoderProtocol(abc.ABC, torch.nn.Module):
    """Protocol for Transformer Decoder.

    Ensures any custom layers are identical to torch.nn.TransformerDecoder.
    """

    @abc.abstractmethod
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
        ...  # noqa: WPS428
