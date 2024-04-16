import abc

import torch


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
        ...
