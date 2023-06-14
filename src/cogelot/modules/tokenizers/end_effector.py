from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, get_args

from cogelot.structures.vima import EndEffector


if TYPE_CHECKING:
    from collections.abc import Mapping


UNK_TOKEN = "[UNK]"  # noqa: S105

END_EFFECTOR_VOCAB = MappingProxyType[str, int](
    {end_effector: idx for idx, end_effector in enumerate(get_args(EndEffector))}
)


class EndEffectorTokenizer:
    """Tokenize end effectors.

    This is simple on purpose, and is a class because it is easier to use with everything else.
    """

    def __init__(self, vocab: Mapping[str, int] = END_EFFECTOR_VOCAB) -> None:
        self.vocab = vocab

    def encode(self, end_effector: str) -> int:
        """Encode an end effector into a token."""
        return self.vocab[end_effector]
