import abc

import torch


class EmbedderProtocol(abc.ABC, torch.nn.Module):
    """Protocol for the Embedder from x-transformers."""

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of the embedder, returning embedded instances."""
        ...


class PassthroughEmbedder(EmbedderProtocol):
    """Embedder that just returns the input and does nothing."""

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        """Just return the input."""
        return torch.zeros_like(x)
