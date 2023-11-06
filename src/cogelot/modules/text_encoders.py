from typing import ClassVar

import torch
from transformers import AutoModel
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5EncoderModel


class T5PromptEncoder(T5EncoderModel):  # type: ignore[misc]
    """Prompt encoder based on T5."""

    authorized_missing_keys: ClassVar[list[str]] = [r"encoder.embed_tokens.weight"]

    def __init__(self, config: T5Config, *, unfreeze_last_n_layers: int = 2) -> None:
        super().__init__(config)

        self.output_dim = self.config.d_model
        self.input_dim = self.config.d_model

        # Even though the VIMA paper claimed it, it was not included in their original code.
        # Freeze all layers except the last n layers
        if unfreeze_last_n_layers > 0:
            blocks_to_freeze = self.encoder.block[:-unfreeze_last_n_layers]
            assert isinstance(blocks_to_freeze, torch.nn.ModuleList)
            for block in blocks_to_freeze:
                for param in block.parameters():
                    param.requires_grad = False


class T5TextEmbedder(torch.nn.Module):
    """Just the embedding layer from T5."""

    def __init__(self, pretrained_model: str) -> None:
        super().__init__()
        model = AutoModel.from_pretrained(pretrained_model)
        embed_weight = model.get_input_embeddings().weight.data
        self._embed_layer = torch.nn.Embedding.from_pretrained(embed_weight)
        self.output_dim = embed_weight.shape[1]
        del model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed the input with T5."""
        x = self._embed_layer(x)
        return x
