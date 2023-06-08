from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast

import torch

from cogelot.data.structures import Assets, ImageType, View
from cogelot.data.token import TextToken, Token, TokenType, VisualToken


if TYPE_CHECKING:
    from cogelot.modules.tokenizers.image import ImageTokenizer
    from cogelot.modules.tokenizers.text import TextTokenizer

T = TypeVar("T", bound=Token)


class MultimodalPromptTokenizer:
    """Tokenize a multimodal prompt."""

    def __init__(
        self,
        text_tokenizer: TextTokenizer,
        image_tokenizer: ImageTokenizer,
        views: list[View],
    ) -> None:
        self.text_tokenizer = text_tokenizer.tokenizer
        self.image_tokenizer = image_tokenizer

        self._views = views
        self._placeholder_names = text_tokenizer.all_placeholders

    def forward_single_prompt(
        self, *, input_text: str, assets: Assets
    ) -> list[TextToken | VisualToken]:
        """Tokenize a single prompt into a list of tokens."""
        tokenized_text = self.tokenize_string(input_text)
        tokens_per_modality = self.split_tokens_by_token_type(tokenized_text)
        visual_tokens = [
            self.convert_text_token_to_visual_token(token, assets=assets)
            for token in tokens_per_modality[TokenType.image]
        ]

        # Sort all the tokens by position
        all_tokens = [*tokens_per_modality[TokenType.text], *visual_tokens]
        all_tokens.sort(key=lambda token: token.index)
        return all_tokens

    @torch.no_grad()
    def tokenize_string(self, text: str) -> list[TextToken]:
        """Tokenize a string of text into tokens."""
        text_encoding = self.text_tokenizer(text)

        token_ids = cast(list[int], text_encoding["input_ids"])
        token_values: list[str] = text_encoding.tokens()

        text_tokens: list[TextToken] = [
            TextToken(token_id=token_id, token=token_value, index=idx)
            for idx, (token_id, token_value) in enumerate(
                zip(token_ids, token_values, strict=True)
            )
        ]

        return text_tokens

    @torch.no_grad()
    def split_tokens_by_token_type(self, tokens: list[T]) -> dict[TokenType, list[T]]:
        """Extract text tokens with placeholders."""
        split_tokens: dict[TokenType, list[T]] = {modality: [] for modality in TokenType}

        for token in tokens:
            if token.token in self._placeholder_names:
                split_tokens[TokenType.image].append(token)
            else:
                split_tokens[TokenType.text].append(token)

        return split_tokens

    @torch.no_grad()
    def convert_text_token_to_visual_token(
        self, token: TextToken, *, assets: Assets
    ) -> VisualToken:
        """Convert text token to visual tokens."""
        if not token.token:
            raise ValueError("Token value is empty.")

        asset = assets.get_asset_from_placeholder(token.token)
        image_per_type_per_view = {
            view: {
                ImageType.rgb: asset.rgb.get_view(view),
                ImageType.segmentation: asset.segm.get_view(view),
            }
            for view in self._views
        }

        visual_token = self.image_tokenizer.create_visual_token_from_images(
            token_position_idx=token.index,
            token_value=token.token,
            image_per_type_per_view=image_per_type_per_view,
            available_object_ids=asset.object_ids,
        )
        return visual_token
