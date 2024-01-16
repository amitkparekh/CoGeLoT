import random
import string

from cogelot.modules.tokenizers.text import LEFT_SYMBOL
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
    for word in words:
        if word.startswith(LEFT_SYMBOL):
            continue

        new_word = _generate_random_characters(length=len(word))
        prompt = prompt.replace(word, new_word, 1)

    return prompt


class VIMAInstanceTransform:
    """Transform VIMA instances by applying a function to them.

    This will allow us to, especially during evaluation, to modify the instance before we actually
    provide it to the environment.
    """

    def __call__(self, instance: VIMAInstance) -> VIMAInstance:
        """Return the instance without transforming it.."""
        raise NotImplementedError


class NullTransform(VIMAInstanceTransform):
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
