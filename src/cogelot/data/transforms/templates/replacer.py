import random

from pydantic import BaseModel

from cogelot.data.transforms.templates.formatter import TemplateFormatter
from cogelot.structures.vima import Task


def extract_keys_from_original(prompt: str, template: str) -> dict[str, str]:
    """Extract keys from the original prompt."""
    # Remove any '.'s from the end before splittings
    words = [word.rstrip(".").rstrip(":") for word in prompt.lower().split(" ")]
    placeholders = [
        placeholder.strip().rstrip(".").rstrip(":") for placeholder in template.lower().split(" ")
    ]

    extracted_key_values = {}
    for placeholder, word in zip(placeholders, words, strict=True):
        if placeholder.startswith("{") and placeholder.endswith("}"):
            extracted_key_values[placeholder[1:-1]] = word

    return extracted_key_values


def is_new_prompt_valid(new_prompt: str, necessary_placeholders: list[str]) -> bool:
    """Check if a new prompt is valid.

    Check that all the placeholders from the original prompt are in the new one. This is to ensure
    that the new prompt is still valid.
    """
    return all(f"{{{placeholder}}}" in new_prompt for placeholder in necessary_placeholders)


class TemplateReplacer(BaseModel):
    """Replace a provided prompt with one generated from templates."""

    _max_attempts = 100
    task: Task
    original_reuse_allowed: bool = True

    templates: list[str]
    key_replacements: dict[str, set[str]]

    def get_template_that_best_matches_prompt(self, original: str) -> str:
        """Find the template that best matches the given prompt.

        We can brute force this by comparing the number of words for each template and the prompt.
        The one with the least difference is the best match.
        """
        # Start by checking the length of the prompt against the task.
        templates_with_correct_length = self._get_templates_with_same_length(original)

        # For each of the possible templates, extract the keys from the prompt for the template and
        # then for each key, ensure that its value is already in the list of possible replacements
        # since that's how they've been made.
        valid_templates = [
            template
            for template in templates_with_correct_length
            if self._do_words_match_with_replacements(template, original)
        ]

        # If there are no templates left, then we have a problem
        if not valid_templates:
            raise RuntimeError(
                "No template with the correct length and keys found. This should not be possible."
            )

        return random.choice(valid_templates)  # noqa: S311

    def choose_random_template(self) -> str:
        """Randomly choose a template."""
        return random.choice(list(self.templates))  # noqa: S311

    def randomly_choose_key_replacements(
        self, keys_from_original: dict[str, str]
    ) -> dict[str, str]:
        """Get key replacements for the given prompt."""
        key_replacements = self._randomly_choose_key_replacements()
        for key, original_value in keys_from_original.items():
            if key not in key_replacements:
                key_replacements[key] = original_value
        return key_replacements

    def generate_new_prompt(
        self,
        original_prompt: str,
        keys_from_original: dict[str, str],
        necessary_placeholders: list[str],
        *,
        skip_key_value_randomisation: bool = False,
    ) -> str:
        """Generate a new prompt."""
        key_replacements = keys_from_original
        if not skip_key_value_randomisation:
            key_replacements = self.randomly_choose_key_replacements(keys_from_original)

        is_valid = False
        counter = 0
        while not is_valid:
            template = self.choose_random_template()
            new_prompt = TemplateFormatter().format(template, **key_replacements)

            is_valid = is_new_prompt_valid(new_prompt, necessary_placeholders)
            # If we don't allow reuse of the original prompt, we need to check
            if original_prompt.lower() == new_prompt.lower() and not self.original_reuse_allowed:
                is_valid = False

            counter += 1
            if counter > self._max_attempts:
                raise RuntimeError(f"Could not generate a valid prompt after {counter} attempts.")

        # HACK: This is a hack. Forgive me. #noqa: FIX004
        new_prompt = new_prompt.replace("{adv}", "").replace("  ", " ")

        return new_prompt

    def _get_templates_with_same_length(self, original: str) -> list[str]:
        """Get all templates with the same length as the original prompt."""
        templates = [
            template
            for template in self.templates
            if len(original.split(" ")) == len(template.split(" "))
        ]

        if not templates:
            raise RuntimeError(
                "No template with the correct length found. This should not be possible."
            )

        return templates

    def _do_words_match_with_replacements(self, template: str, original: str) -> bool:
        """Do the words in the original match the replacements that go in that position.

        For each of the possible templates, extract the keys from the prompt for the template and
        then for each key, ensure that its value is already in the list of possible replacements
        since that's how they've been made.
        """
        keys_from_template = extract_keys_from_original(original, template)

        for key, extracted_value in keys_from_template.items():
            if key in self.key_replacements and extracted_value not in self.key_replacements[key]:
                return False

        return True

    def _randomly_choose_key_replacements(self) -> dict[str, str]:
        """Randomly choose key replacements."""
        return {
            key: random.choice(list(alternatives))  # noqa: S311
            for key, alternatives in self.key_replacements.items()
        }
