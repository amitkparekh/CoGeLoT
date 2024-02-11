import random
import string
from functools import cached_property
from typing import cast

from cogelot.data.transforms.base import VIMAInstanceTransform
from cogelot.modules.tokenizers.text import LEFT_SYMBOL, TextTokenizer
from cogelot.structures.vima import VIMAInstance

ALPHABET = list(string.ascii_uppercase + string.ascii_lowercase)


class WordTooLongError(Exception):
    """Raise when the word is too long."""


def _generate_random_characters(*, length: int) -> str:
    """Generate a string of random characters."""
    return "".join(random.choices(ALPHABET, k=length))  # noqa: S311


def convert_language_prompt_to_gobbledygook(prompt: str) -> str:
    """Convert language in prompt to gobbledygook (i.e., nonsense), keeping the word count."""
    # Break up the prompt into words
    words = prompt.split(" ")

    # For any word that is not the special token, replace it with nonsense
    for idx, word in enumerate(words):
        if word.startswith(LEFT_SYMBOL):
            continue

        words[idx] = _generate_random_characters(length=len(word))

    return " ".join(words)


class GobbledyGookPromptWordTransform(VIMAInstanceTransform):
    """For each word in the prompt, randomise the characters but keep the word length."""

    def __call__(self, instance: VIMAInstance) -> VIMAInstance:
        """Replace the words with gobbledygook."""
        return instance.model_copy(
            deep=True, update={"prompt": convert_language_prompt_to_gobbledygook(instance.prompt)}
        )


class GobbledyGookPromptTokenTransform(VIMAInstanceTransform):
    """Randomise the tokens in the prompt for other tokens in the vocabulary."""

    space_token_id = 103

    def __init__(self, text_tokenizer: TextTokenizer, *, timeout: int = 10) -> None:
        self.text_tokenizer = text_tokenizer

        self.encode = self.text_tokenizer.tokenizer.encode
        self.decode = self.text_tokenizer.tokenizer.decode
        self.tokenize = self.text_tokenizer.tokenizer

        self._timeout = timeout

        # We are going to have a memory of previously valid token sequences in case we come across
        # a timeout. In these situations, we can grab from the list and only if one doesn't exist
        # there, we raise the error.
        self._tokenized_word_memory: dict[tuple[int, ...], set[tuple[int, ...]]] = {}

    def __call__(self, instance: VIMAInstance) -> VIMAInstance:
        """Replace the words with gobbledygook."""
        # Tokenize each word in the prompt separately
        token_ids_per_word = cast(
            list[list[int]],
            self.tokenize(instance.prompt.split(" "), add_special_tokens=False)["input_ids"],
        )

        # For each word in the prompt, randomise the token IDs AND ensure that when the prompt is
        # re-tokenized, it will be the exact same length as before.
        new_prompt_tokens: list[tuple[int, ...]] = [
            self._get_new_token_sequence_for_word(tuple(tokenized_word))
            for tokenized_word in token_ids_per_word
        ]

        new_prompt_tokens = self._randomise_order_of_non_special_words(new_prompt_tokens)

        new_prompt = " ".join(self.text_tokenizer.tokenizer.batch_decode(new_prompt_tokens))
        return instance.model_copy(deep=True, update={"prompt": new_prompt})

    def _randomise_order_of_non_special_words(
        self, tokenized_words: list[tuple[int, ...]]
    ) -> list[tuple[int, ...]]:
        """Randomise the order of any token that does not contain 'special' characters.

        Note: 103 is the space token in T5.
        """
        shuffled_list = tokenized_words.copy()

        # Determine which words can change their positions
        locked_positions = []
        unlocked_words = []
        for word_index, tokenized_word in enumerate(tokenized_words):
            has_special_token = bool(
                set(tokenized_word) - set(self._non_special_token_ids) - {self.space_token_id}
            )
            if not has_special_token:
                unlocked_words.append(tokenized_word)
            if has_special_token:
                locked_positions.append(word_index)

        # Randomise the order of the unlocked positions
        random.shuffle(unlocked_words)

        # Put the unlocked words back into the shuffled list
        for idx, _ in enumerate(shuffled_list):
            if idx not in locked_positions:
                shuffled_list[idx] = unlocked_words.pop(0)

        return shuffled_list

    def _get_new_token_sequence_for_word(self, tokenized_word: tuple[int, ...]) -> tuple[int, ...]:
        """Randomise the token IDs and verify the new sequence is the same length as the old."""
        # Add the tokenized word to the memory
        if tokenized_word not in self._tokenized_word_memory:
            self._tokenized_word_memory[tokenized_word] = set()

        # Keep trying to randomise the token sequence until we have one that is the same length as
        # the original tokenized word
        iteration_count = 0
        while iteration_count <= self._timeout:
            try:
                random_token_sequence = self._randomise_token_ids_in_sequence(tokenized_word)
            except WordTooLongError:
                iteration_count += 1
            else:
                break

        # If we have a timeout, we are going to try and find a random token sequence from the
        # memory and hope we have one
        if iteration_count > self._timeout:
            try:
                random_token_sequence = self._get_random_alternative_from_memory(tokenized_word)
            except (KeyError, IndexError) as err:
                raise RuntimeError(
                    "Could not find a random token sequence that is the same length as the "
                    "original token word."
                ) from err

        # Add the new token sequence to the memory
        self._tokenized_word_memory[tokenized_word].add(random_token_sequence)

        return random_token_sequence

    def _randomise_token_ids_in_sequence(self, sequence: tuple[int, ...]) -> tuple[int, ...]:
        """Randomise the token IDs in a sequence.

        If a sequence has a special token, it is likely going to be incredibly small and we can
        return it on its own to avoid any issues with the length of the sequence or trying to
        regenerate around it.
        """
        # Return the sequence as is if it has any special tokens in it
        if any(token not in self._non_special_token_ids for token in sequence):
            return sequence

        new_token_sequence = []
        re_tokenized_sequence_len = 0

        while re_tokenized_sequence_len != len(sequence):
            # Pick a random token, non-special, to add to the sequence
            random_token = random.choice(self._non_special_token_ids)  # noqa: S311
            new_token_sequence.append(random_token)

            # Token sequence length after re-tokenizing
            re_tokenized_sequence_len = len(
                self.encode(self.decode(tuple(new_token_sequence)), add_special_tokens=False)
            )

            if re_tokenized_sequence_len > len(sequence):
                raise WordTooLongError

        return tuple(new_token_sequence)

    @cached_property
    def _non_special_token_ids(self) -> list[int]:
        """Get the set of token ids to choose from."""
        all_non_special = list(
            set(range(self.text_tokenizer.tokenizer.vocab_size))
            - set(self.text_tokenizer.tokenizer.all_special_ids)
            - set(self.text_tokenizer.tokenizer.additional_special_tokens_ids)
            - set(self.text_tokenizer.placeholder_token_ids)
            # Changing this token can cause a lot of problems in the new token creation, so we are
            # not going to be replacing it.
            - set(self.encode(".", add_special_tokens=False))
        )
        return all_non_special

    def _get_random_alternative_from_memory(
        self, tokenized_word: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Get a random alternative from the memory.

        Is the data structure swapping silly? Yes.
        """
        return random.choice(tuple(self._tokenized_word_memory[tokenized_word]))  # noqa: S311
