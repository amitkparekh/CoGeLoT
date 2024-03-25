import torch
from einops import rearrange, repeat

from cogelot.nn.decoders.interfaces import TransformerDecoderProtocol
from vima import nn as vnn


class VIMADecoder(TransformerDecoderProtocol):
    """Their XAttnGPT Decoder but wrapped in our interface.

    We are abstracting out the logic from their Policy so we can more easily swap between different
    modules for experiments.
    """

    def __init__(self, vima_xattn_gpt: vnn.XAttnGPT) -> None:
        super().__init__()
        self._vima_xattn_gpt = vima_xattn_gpt

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
        if tgt_mask is not None:
            raise ValueError("`tgt_mask` is not used and nothing should be passed in to it.")
        if memory_mask is not None:
            raise ValueError("`memory_mask` is not used and nothing should be passed in to it.")

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

        # Create the position ids from the mask (just how they do in the Policy)
        position_ids = torch.cumsum(tgt_key_padding_mask, dim=1)
        position_ids = position_ids.long()
        prompt_position_ids = torch.cumsum(memory_key_padding_mask, dim=1)

        tokens_out = self._vima_xattn_gpt(
            obs_action_tokens=tgt,
            prompt_tokens=memory,
            prompt_mask=memory_key_padding_mask,
            obs_action_masks=tgt_key_padding_mask,
            obs_action_position_ids=position_ids,
            prompt_position_ids=prompt_position_ids,
            batch_first=True,
        )

        return tokens_out


class VIMAGPTDecoderOnly(TransformerDecoderProtocol):
    """Their GPT decoder but wrapped in our interface."""

    def __init__(self, vima_hf_gpt: vnn.HFGPT) -> None:
        super().__init__()
        self._vima_hf_gpt = vima_hf_gpt
        self.prompt_sep_token = torch.nn.Parameter(torch.zeros(self._vima_hf_gpt.lm.config.n_embd))

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
        if tgt_mask is not None:
            raise ValueError("`tgt_mask` is not used and nothing should be passed in to it.")
        if memory_mask is not None:
            raise ValueError("`memory_mask` is not used and nothing should be passed in to it.")

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

        tokens = torch.cat(
            [
                memory,
                repeat(
                    rearrange(self.prompt_sep_token, "d -> () () d"),
                    "1 s d -> b s d",
                    b=memory.size(0),
                ),
                tgt,
            ],
            dim=1,
        )
        mask = torch.cat(
            [
                memory_key_padding_mask,
                torch.ones(
                    memory_key_padding_mask.size(0),
                    1,
                    dtype=memory_key_padding_mask.dtype,
                    device=memory_key_padding_mask.device,
                ),
                tgt_key_padding_mask,
            ],
            dim=1,
        )
        position_ids = torch.cumsum(mask, dim=1).long() - 1
        transformer_output = self._vima_hf_gpt(
            x=tokens, custom_mask=mask, position_ids=position_ids, batch_first=True
        )

        # Slice off the memory so we just return the obs and actions
        tokens_out = transformer_output[:, -tgt.size(1) :]
        return tokens_out
