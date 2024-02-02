import random
from typing import TYPE_CHECKING

from pydantic import BaseModel
from tqdm.contrib.itertools import product

from cogelot.data.templating.formatter import TemplateFormatter
from cogelot.structures.vima import Task

if TYPE_CHECKING:
    from collections.abc import Generator


def _extract_keys_from_original(prompt: str, template: str) -> dict[str, str]:
    """Extract keys from the original prompt."""
    # Remove any '.'s from the end before splittings
    words = prompt.rstrip(".").split(" ")
    placeholders = template.split(" ")

    extracted_key_values = {}
    for placeholder, word in zip(placeholders, words, strict=True):
        if placeholder.startswith("{") and placeholder.endswith("}"):
            extracted_key_values[placeholder[1:-1]] = word

    return extracted_key_values


def _is_new_prompt_valid(new_prompt: dict[str, str], original_prompt_keys: dict[str, str]) -> bool:
    """Check if a new prompt is valid.

    Check that all the placeholders from the original prompt are in the new one This is to ensure
    that the new prompt is still valid.
    """
    original_placeholders = set(original_prompt_keys.keys())
    new_placeholders = set(new_prompt.keys())
    return original_placeholders.issubset(new_placeholders)


class TemplateReplacer(BaseModel):
    """Replace a provided prompt with one generated from templates."""

    _max_attempts = 100
    task: Task
    original_reuse_allowed: bool = True

    templates: set[str]
    key_replacements: dict[str, set[str]]

    def __call__(self, prompt: str) -> str:
        """Regenerate the prompt into a new one."""
        return self.regenerate_original_prompt(prompt)

    def regenerate_original_prompt(self, prompt: str) -> str:
        """Regenerate an original prompt into a new one."""
        keys_from_original = self._get_keys_from_original(prompt)
        key_replacements = self._get_new_key_replacements(keys_from_original)

        is_valid = False
        counter = 0
        while not is_valid:
            new_prompt = self._generate_prompt(key_replacements)
            is_valid = _is_new_prompt_valid(key_replacements, keys_from_original)

            # If we don't allow reuse of the original prompt, we need to check
            if new_prompt == prompt and not self.original_reuse_allowed:
                is_valid = False

            counter += 1

            if counter > self._max_attempts:
                raise RuntimeError(f"Could not generate a valid prompt after {counter} attempts.")

        return new_prompt

    def get_all_possible_templates(self, original_prompt: str) -> set[str]:
        """Get all possible templates for the given prompt."""
        formatter = TemplateFormatter()
        keys_from_original = self._get_keys_from_original(original_prompt)
        key_replacements = self._get_all_possible_key_replacements(keys_from_original)
        all_template_key_combinations: Generator[tuple[str, ...], None, None] = product(
            self.templates,
            *key_replacements.values(),
            desc=f"Generating templates for {self.task}",
        )

        all_generated_templates = set()
        for template, *replacements in all_template_key_combinations:
            all_generated_templates.add(
                formatter.format(
                    template, **dict(zip(key_replacements.keys(), replacements, strict=False))
                )
            )
        return all_generated_templates

    def _get_keys_from_original(self, original: str) -> dict[str, str]:
        """Get the keys used in the original prompt."""
        return _extract_keys_from_original(
            original, self._get_template_that_best_matches_prompt(original)
        )

    def _get_template_that_best_matches_prompt(self, original: str) -> str:
        """Find the template that best matches the given prompt.

        We can brute force this by comparing the number of words for each template and the prompt.
        The one with the least difference is the best match.
        """
        original_words = original.split(" ")
        best_match: str = ""
        best_match_difference: int = 1000

        for template in self.templates:
            template_words = template.split(" ")
            difference = abs(len(original_words) - len(template_words))
            if difference < best_match_difference:
                best_match = template
                best_match_difference = difference

            # If the length differences are the same, we can stop here
            if best_match_difference == 0:
                break

        return best_match

    def _randomly_choose_template(self) -> str:
        """Randomly choose a template."""
        return random.choice(list(self.templates))  # noqa: S311

    def _generate_prompt(self, key_replacements: dict[str, str]) -> str:
        """Generate a prompt."""
        formatter = TemplateFormatter()
        template = self._randomly_choose_template()
        return formatter.format(template, **key_replacements)

    def _randomly_choose_key_replacements(self) -> dict[str, str]:
        """Randomly choose key replacements."""
        return {
            key: random.choice(list(alternatives))  # noqa: S311
            for key, alternatives in self.key_replacements.items()
        }

    def _get_new_key_replacements(self, keys_from_original: dict[str, str]) -> dict[str, str]:
        """Get key replacements for the given prompt."""
        key_replacements = self._randomly_choose_key_replacements()
        for key, original_value in keys_from_original.items():
            if key not in key_replacements:
                key_replacements[key] = original_value
        return key_replacements

    def _get_all_possible_key_replacements(
        self, keys_from_original: dict[str, str]
    ) -> dict[str, set[str]]:
        """Get a list of all possible key replacements from the original prompt."""
        key_replacements = self.key_replacements
        for key, original_value in keys_from_original.items():
            if key not in key_replacements:
                key_replacements[key] = {original_value}

        return key_replacements
