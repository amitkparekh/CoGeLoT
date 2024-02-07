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


class TransformerDecoderGreedyGenerateWrapper(TransformerDecoderProtocol):
    """During inference, wrap the forward call with greedy generation."""

    def __init__(
        self, decoder: TransformerDecoderProtocol, *, num_tokens_to_generate_per_timestep: int = 1
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self._num_tokens_to_generate_per_timestep = num_tokens_to_generate_per_timestep

    @torch.no_grad()
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Greedily generate multiple tokens until you reach the desired number."""
        # For a number of generated tokens, the desired output sequence length is
        # (seq_length + num_tokens_to_generate_per_timestep - 1), so we keep going until we have that. Since the
        # `range` function starts from 0, we can iterate from the starting length to
        # (starting length + num_tokens_to_generate_per_timestep)
        desired_seq_length = tgt.size(1) + self._num_tokens_to_generate_per_timestep

        # Shape (batch size, seq length, dim)
        transformer_output = self.decoder(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
        )
        while transformer_output.size(1) != desired_seq_length:
            # Take the last token length and update the tgt tensor so it is one step longer. We
            # also need to extend the masks if they are not None
            tgt = torch.cat([tgt, transformer_output[:, -1:]], dim=1)

            if tgt_mask is not None:
                tgt_mask = torch.cat([tgt_mask, tgt_mask[:, -1:]], dim=1)
            if tgt_key_padding_mask is not None:
                tgt_key_padding_mask = torch.cat(
                    [tgt_key_padding_mask, tgt_key_padding_mask[:, -1:]], dim=1
                )
            transformer_output = self.decoder(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
            )

        return transformer_output
