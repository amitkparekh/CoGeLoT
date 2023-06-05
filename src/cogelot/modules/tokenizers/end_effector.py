from __future__ import annotations

from tokenizers import Tokenizer
from tokenizers.models import WordLevel


UNK_TOKEN = "[UNK]"  # noqa: S105

END_EFFECTOR_VOCAB = {
    "suction": 0,
    "spatula": 1,
}


def create_end_effector_tokenizer(
    end_effector_vocab: dict[str, int] = END_EFFECTOR_VOCAB, unk_token: str = UNK_TOKEN
) -> Tokenizer:
    """Create a tokenizer for the end effector."""
    return Tokenizer(WordLevel(vocab=end_effector_vocab, unk_token=unk_token))
