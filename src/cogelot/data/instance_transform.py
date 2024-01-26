import random
import string
from functools import cached_property
from typing import cast

from cogelot.modules.tokenizers.text import LEFT_SYMBOL, TextTokenizer
from cogelot.structures.vima import VIMAInstance

ALPHABET = list(string.ascii_uppercase + string.ascii_lowercase)


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


class VIMAInstanceTransform:
    """Transform VIMA instances by applying a function to them.

    This will allow us to, especially during evaluation, to modify the instance before we actually
    provide it to the environment.
    """

    def __call__(self, instance: VIMAInstance) -> VIMAInstance:
        """Return the instance without transforming it.."""
        raise NotImplementedError


class NoopTransform(VIMAInstanceTransform):
    """Do not transform the instance."""

    def __call__(self, instance: VIMAInstance) -> VIMAInstance:
        """Return the instance without transforming it.."""
        return instance


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

    def __call__(self, instance: VIMAInstance) -> VIMAInstance:
        """Replace the words with gobbledygook."""
        # Tokenize each word in the prompt separately
        token_ids_per_word = cast(
            list[list[int]],
            self.tokenize(instance.prompt.split(" "), add_special_tokens=False)["input_ids"],
        )

        # For each word in the prompt, randomise the token IDs AND ensure that when the prompt is
        # re-tokenized, it will be the exact same length as before.
        new_prompt_tokens: list[list[int]] = [
            self._get_new_token_sequence_for_word(tokenized_word)
            for tokenized_word in token_ids_per_word
        ]

        new_prompt_tokens = self._randomise_order_of_non_special_words(new_prompt_tokens)

        new_prompt = " ".join(self.text_tokenizer.tokenizer.batch_decode(new_prompt_tokens))
        return instance.model_copy(deep=True, update={"prompt": new_prompt})

    def _randomise_order_of_non_special_words(
        self, tokenized_words: list[list[int]]
    ) -> list[list[int]]:
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

    def _get_new_token_sequence_for_word(self, tokenized_word: list[int]) -> list[int]:
        """Randomise the token IDs and verify the new sequence is the same length as the old."""
        iteration_count = 0
        random_token_sequence: list[int] = []

        while len(tokenized_word) != len(random_token_sequence):
            random_token_sequence = self._randomise_token_ids_in_sequence(tokenized_word)
            random_token_sequence = self.encode(
                self.decode(random_token_sequence), add_special_tokens=False
            )
            iteration_count += 1
            if iteration_count > self._timeout:
                raise RuntimeError(
                    "Could not find a random token sequence that is the same length as the "
                    "original token word."
                )

        return random_token_sequence

    def _randomise_token_ids_in_sequence(self, sequence: list[int]) -> list[int]:
        """Randomise the token IDs in a sequence with some alphabet characters.

        Sometimes, words that have only a single token are special and replacing them with anything
        else will result in their length being > 1. This then results in a valid solution not able
        to be found, and we don't want this. As a result, we always return the first token in the
        sequence as it is, even if its a sequence of length 1, so that it is always the same
        length.

        This is fine because in the grand scheme of things, having one word that is "normal" is not
        going to be a drastic problem.
        """
        randomised_sequence = [
            random.choice(self._alphabet_token_ids)  # noqa: S311
            if token in self._non_special_token_ids
            else token
            for token in sequence[1:]
        ]
        return [sequence[0], *randomised_sequence]

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

    @cached_property
    def _alphabet_token_ids(self) -> list[int]:
        """Get all token IDs for the alphabet."""
        token_ids = self.text_tokenizer.tokenizer.convert_tokens_to_ids(ALPHABET)
        assert isinstance(token_ids, list)
        return token_ids
