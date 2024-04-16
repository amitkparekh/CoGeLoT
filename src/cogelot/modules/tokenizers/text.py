from collections.abc import Iterable
from functools import cached_property
from typing import Any

from pydantic import BaseModel
from tokenizers import AddedToken
from transformers import AutoTokenizer

LEFT_SYMBOL = "{"
RIGHT_SYMBOL = "}"


class AssetPlaceholderToken(BaseModel):
    """Placeholder token.

    If there is more than one repeat needed (e.g., prefixed with "_1" etc), then set the `max_idx`
    to the maximum index number that will exist.
    """

    asset_name: str
    max_idx: int | None = None
    left_symbol: str = LEFT_SYMBOL
    right_symbol: str = RIGHT_SYMBOL

    @property
    def all_placeholders(self) -> list[str]:
        """Return the list of placeholder tokens for this template."""
        token_list = [self.asset_name]
        if self.max_idx:
            tokens = [f"{self.asset_name}_{idx}" for idx in range(self.max_idx + 1)]
            token_list.extend(tokens)

        return [f"{self.left_symbol}{token}{self.right_symbol}" for token in token_list]

    @property
    def all_added_tokens(self) -> list[AddedToken]:
        """Return a list of all placeholders as `AddedToken`s."""
        added_token_list = [
            AddedToken(token, single_word=True, lstrip=False, rstrip=False, normalized=True)
            for token in self.all_placeholders
        ]
        return added_token_list


PLACEHOLDER_TOKENS = (
    AssetPlaceholderToken(asset_name="base_obj", max_idx=10),
    AssetPlaceholderToken(asset_name="dragged_obj", max_idx=10),
    AssetPlaceholderToken(asset_name="swept_obj"),
    AssetPlaceholderToken(asset_name="bounds"),
    AssetPlaceholderToken(asset_name="constraint"),
    AssetPlaceholderToken(asset_name="scene"),
    AssetPlaceholderToken(asset_name="demo_blicker_obj", max_idx=10),
    AssetPlaceholderToken(asset_name="demo_less_blicker_obj", max_idx=10),
    AssetPlaceholderToken(asset_name="start_scene", max_idx=10),
    AssetPlaceholderToken(asset_name="end_scene", max_idx=10),
    AssetPlaceholderToken(asset_name="before_twist", max_idx=10),
    AssetPlaceholderToken(asset_name="after_twist", max_idx=10),
    AssetPlaceholderToken(asset_name="frame", max_idx=10),
    AssetPlaceholderToken(asset_name="ring"),
    AssetPlaceholderToken(asset_name="hanoi_stand"),
)


class TextTokenizer:
    """Text tokenizer for text inputs."""

    def __init__(
        self,
        pretrained_model: str,
        placeholder_tokens: Iterable[AssetPlaceholderToken] = PLACEHOLDER_TOKENS,
        **kwargs: Any,
    ) -> None:
        """Create a tokenizer from something."""
        # Instantiate from the pretrained model
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model, **kwargs)

        # Add all the placeholder tokens to the tokenizer
        for token in placeholder_tokens:
            tokenizer.add_tokens(token.all_added_tokens)

        self.tokenizer = tokenizer
        self.placeholder_tokens = placeholder_tokens

    @cached_property
    def all_placeholders(self) -> set[str]:
        """Get the list of all the placeholders used from the tokens."""
        return {
            token
            for placeholder in self.placeholder_tokens
            for token in placeholder.all_placeholders
        }

    @cached_property
    def placeholder_token_ids(self) -> set[int]:
        """Get the list of all the token IDs that connect to placeholders."""
        token_ids = self.tokenizer.convert_tokens_to_ids(list(self.all_placeholders))
        assert isinstance(token_ids, list)
        return set(token_ids)
