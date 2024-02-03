import random

from pydantic import BaseModel

from cogelot.structures.vima import Task


def extract_keys_from_original(prompt: str, template: str) -> dict[str, str]:
    """Extract keys from the original prompt."""
    extracted_key_values = {}
    # Remove any '.'s from the end before splittings
    words = prompt.lower().rstrip(".").split(" ")
    placeholders = template.split(" ")

    for placeholder, word in zip(placeholders, words, strict=True):
        if placeholder.startswith("{") and placeholder.rstrip(".").endswith("}"):
            extracted_key_values[placeholder.rstrip(".")[1:-1]] = word.rstrip(".")

    return extracted_key_values


def is_new_prompt_valid(new_prompt: str, necessary_placeholders: list[str]) -> bool:
    """Check if a new prompt is valid.

    Check that all the placeholders from the original prompt are in the new one. This is to ensure
    that the new prompt is still valid.
    """
    return all(f"{{{placeholder}}}" in new_prompt for placeholder in necessary_placeholders)

    # new_placeholders = {
    #     placeholder[1:-1]
    #     for placeholder in template.split(" ")
    #     if placeholder.startswith("{") and placeholder.endswith("}")
    # }
    # original_placeholders = set(necessary_placeholders)
    # return original_placeholders.issubset(new_placeholders)


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
        for template in templates_with_correct_length:
            if not self._do_words_match_with_replacements(template, original):
                templates_with_correct_length.remove(template)

        # If there are no templates left, then we have a problem
        if not templates_with_correct_length:
            raise RuntimeError(
                "No template with the correct length and keys found. This should not be possible."
            )

        return random.choice(templates_with_correct_length)  # noqa: S311

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
