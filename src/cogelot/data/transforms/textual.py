from typing import ClassVar

from cogelot.data.transforms.base import VIMAInstanceTransform
from cogelot.data.transforms.reword import (
    CONTAIN_PREPOSITION,
    CONTAINER_NOUN,
    EXCEEDING_CONSTRAINTS,
    FINALLY_ALTERNATIVES,
    GENERIC_PLURAL_NOUNS,
    GENERIC_SINGULAR_NOUNS,
    MOVED_VERBS,
    PLACING_VERBS,
    PRECEEDING_ADJECTIVES,
    REARRANGE_PREPOSITIONS,
    REARRANGE_VERBS,
    RESTORE_WORDS,
    SCENE_WORDS,
    STARTING_ALTERNATIVES,
    SWEEP_VERBS,
    TOUCHING_CONSTRAINTS,
)
from cogelot.data.transforms.templates.formatter import DefaultFormatter
from cogelot.data.transforms.templates.replacer import TemplateReplacer, extract_keys_from_original
from cogelot.structures.vima import Task, VIMAInstance


class TextualDescriptionTransform(VIMAInstanceTransform):
    """Convert any visual placeholders to natural language descriptions, where relevant."""

    # These tasks should be avoided for this transform because they don't make ecological sense.
    tasks_to_avoid: ClassVar[set[Task]] = {
        Task.novel_adj,
        Task.follow_order,
        Task.twist,
        Task.rearrange,
        Task.rearrange_then_restore,
        Task.novel_adj_and_noun,
        Task.same_texture,
        Task.same_shape,
        Task.scene_understanding,
    }

    def __init__(self) -> None:
        self._formatter = DefaultFormatter()

        self.replacers = {
            Task.pick_in_order_then_restore: TemplateReplacer(
                task=Task.pick_in_order_then_restore,
                original_reuse_allowed=False,
                templates=[
                    # originals
                    "{placing_verb} {object1} {contain_preposition} {container1}. {finally_alternative} {restore_word} it {contain_preposition2} {article1} {starting_alternative} {container_noun}",
                    "{placing_verb} {object1} {contain_preposition} {container1} then {container2}. {finally_alternative} {restore_word} it {contain_preposition2} {article1} {starting_alternative} {container_noun}",
                    # Alternatives
                    "{placing_verb} the {object1} {contain_preposition} the {container1}. {finally_alternative} {restore_word} it {contain_preposition2} {article1} {starting_alternative} {container_noun}",
                    "{placing_verb} the {object1} {contain_preposition} the {container1} and then the {container2}. {finally_alternative} {restore_word} it {contain_preposition2} {article1} {starting_alternative} {container_noun}",
                ],
                key_replacements={
                    "article1": {"the", "its"},
                    "placing_verb": PLACING_VERBS,
                    "contain_preposition": CONTAIN_PREPOSITION,
                    "contain_preposition2": CONTAIN_PREPOSITION,
                    "finally_alternative": FINALLY_ALTERNATIVES,
                    "restore_word": RESTORE_WORDS,
                    "starting_alternative": STARTING_ALTERNATIVES,
                    "container_noun": CONTAINER_NOUN,
                },
            ),
            Task.novel_noun: TemplateReplacer(
                task=Task.novel_noun,
                original_reuse_allowed=False,
                templates=[
                    "This is a {novel_noun1} {object1}. This is a {novel_noun2} {object2}. {placing_verb} a {novel_noun1} {contain_preposition} a {novel_noun2}.",
                    "A {novel_noun1} is a {object1}. A {novel_noun2} is a {object2}. {placing_verb} a {novel_noun1} {contain_preposition} a {novel_noun2}.",
                    "A {novel_noun1} is a {object1}. A {novel_noun2} is a {object2}. {placing_verb} the {novel_noun1} {contain_preposition} the {novel_noun2}.",
                ],
                key_replacements={
                    "rearrange_verb": REARRANGE_VERBS,
                    "preposition": REARRANGE_PREPOSITIONS,
                    "plural_noun": GENERIC_PLURAL_NOUNS,
                    "scene_word": SCENE_WORDS,
                    "restore_word": RESTORE_WORDS,
                    "placing_verb": PLACING_VERBS,
                    "contain_preposition": CONTAIN_PREPOSITION,
                },
            ),
            Task.sweep_without_exceeding: TemplateReplacer(
                task=Task.sweep_without_exceeding,
                original_reuse_allowed=False,
                templates=[
                    "{sweep_verb} {quantity} {object1} {preposition} {bounds} without {exceeding_constraint} {constraint}",
                    "{sweep_verb} {quantity} {object1} {preposition} the {bounds} without {exceeding_constraint} the {constraint}",
                ],
                key_replacements={
                    "sweep_verb": SWEEP_VERBS,
                    "preposition": CONTAIN_PREPOSITION,
                    "exceeding_constraint": EXCEEDING_CONSTRAINTS,
                },
            ),
            Task.sweep_without_touching: TemplateReplacer(
                task=Task.sweep_without_touching,
                original_reuse_allowed=False,
                templates=[
                    "{sweep_verb} {quantity} {object1} {preposition} {bounds} without {touching_constraint} {constraint}",
                    "{sweep_verb} {quantity} {object1} {preposition} the {bounds} without {touching_constraint} the {constraint}",
                ],
                key_replacements={
                    "sweep_verb": SWEEP_VERBS,
                    "preposition": CONTAIN_PREPOSITION,
                    "touching_constraint": TOUCHING_CONSTRAINTS,
                },
            ),
            Task.manipulate_old_neighbor: TemplateReplacer(
                task=Task.manipulate_old_neighbor,
                original_reuse_allowed=False,
                templates=[
                    "{preceeding_adjective} {placing_verb} {object1} {contain_preposition} {object2} then {placing_verb2} {article1} {singular_noun} that was previously at its {direction} into the same {object2}",
                    "{preceeding_adjective} {placing_verb} the {object1} {contain_preposition} the {object2} and then {placing_verb2} {article1} {singular_noun} that was previously at its {direction} into the same {object2}",
                ],
                key_replacements={
                    "placing_verb": PLACING_VERBS,
                    "placing_verb2": PLACING_VERBS,
                    "preceeding_adjective": PRECEEDING_ADJECTIVES,
                    "contain_preposition": CONTAIN_PREPOSITION,
                    "contain_preposition2": CONTAIN_PREPOSITION,
                    "article1": {"the"},
                    "singular_noun": GENERIC_SINGULAR_NOUNS,
                    "container_noun": CONTAINER_NOUN,
                    "moved_verb": MOVED_VERBS,
                },
            ),
        }

    def __call__(self, instance: VIMAInstance) -> VIMAInstance:
        """Process the instance."""
        if instance.task in self.tasks_to_avoid:
            return instance

        placeholder_replacements = {
            placeholder: prompt_asset.as_natural_language
            for placeholder, prompt_asset in instance.prompt_assets.as_dict.items()
            if prompt_asset.as_natural_language
        }

        prompt = self._fix_word_order(instance)
        prompt = self._formatter.format(prompt, **placeholder_replacements)

        return instance.model_copy(deep=True, update={"prompt": prompt})

    def _fix_word_order(self, instance: VIMAInstance) -> str:
        """Fix the word order in the prompt."""
        if instance.task not in self.replacers:
            return instance.prompt

        original_template = self.replacers[instance.task].get_template_that_best_matches_prompt(
            instance.prompt
        )
        keys_from_original = extract_keys_from_original(instance.prompt, original_template)

        new_prompt = self.replacers[instance.task].generate_new_prompt(
            instance.prompt,
            keys_from_original,
            list(instance.prompt_assets.as_dict.keys()),
            skip_key_value_randomisation=True,
        )

        return new_prompt
