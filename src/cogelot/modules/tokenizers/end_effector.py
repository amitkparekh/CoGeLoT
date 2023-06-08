from __future__ import annotations

from typing import get_args

from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from cogelot.data.constants import EndEffector


UNK_TOKEN = "[UNK]"  # noqa: S105

END_EFFECTOR_VOCAB = {end_effector: idx for idx, end_effector in enumerate(get_args(EndEffector))}


def create_end_effector_tokenizer(
    end_effector_vocab: dict[str, int] = END_EFFECTOR_VOCAB, unk_token: str = UNK_TOKEN
) -> Tokenizer:
    """Create a tokenizer for the end effector."""
    return Tokenizer(WordLevel(vocab=end_effector_vocab, unk_token=unk_token))
