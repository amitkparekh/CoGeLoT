import torch
from x_transformers import ContinuousTransformerWrapper

from cogelot.nn.decoders.interfaces import TransformerDecoderProtocol


class XTransformerDecoder(TransformerDecoderProtocol, ContinuousTransformerWrapper):  # type: ignore[misc]
    """XTransformers wrapper for the decoder.

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

        # Create the position ids from the mask (just how they do in the Policy)
        position_ids = (torch.cumsum(tgt_key_padding_mask, dim=1) - 1).long()

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