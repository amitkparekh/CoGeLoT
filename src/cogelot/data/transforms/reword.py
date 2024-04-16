from contextlib import suppress

import more_itertools

from cogelot.data.transforms.base import VIMAInstanceTransform
from cogelot.data.transforms.templates.formatter import TemplateFormatter
from cogelot.data.transforms.templates.replacer import (
    TemplateReplacer,
    extract_keys_from_original,
)
from cogelot.structures.vima import Task, VIMAInstance

ANGLE_UNITS = {"degrees"}
LIFTING_VERBS = {"pick up", "take", "lift", "grab"}
PLACING_VERBS = {"put", "place", "move", "position", "drop", "store", "stick", "set"}
MOVING_VERBS = {"move", "relocate", "shift", "place"}
SWEEP_VERBS = {
    "put",
    "place",
    "position",
    "sweep",
    "push",
    "shove",
    "move",
    *MOVING_VERBS,
}
STACK_VERBS = {"stack", "place", "put", "rearrange"}
ROTATING_VERBS = {"rotate", "turn", "spin", "pivot", "swivel"}
REARRANGE_VERBS = {"rearrange", "reposition", "reorder", "reorganize"}
HOLDING_VERBS = {"hold", "contain", "have"}
DOING_VERB = {"make"}
FOLLOW_VERBS = {"follow", "copy", "imitate"}
MOVED_VERBS = {"moved", "relocated", "shifted", "placed"}
ROTATING_ALTERNATIVES = {
    "rotating",
    "turning",
    "spinning",
    "placing",
    "moving",
    "positioning",
}
PRECEEDING_ADJECTIVES = {"first", "start by", "begin by", "initially"}

ARTICLE = {"the", "one"}
SINGULAR_ARTICLE = {"the", "one"}
PREPOSITION_ARTICLE = {"the", "of the"}

PLURAL_PRONOUNS = {"all", "all the", "the"}
PREPOSITIONS = {
    "onto",
    "into",
    "on top of",
    "within the confines of",
    "on",
}
ASSOCIATION_PREPOSITIONS = {"with"}
CONTAIN_PREPOSITION = {"in", "inside", "within", "inside of", "into"}
REARRANGE_PREPOSITIONS = {"into", "to"}
SINGULAR_PRONOUNS = {"it"}
GENERIC_SINGULAR_NOUNS = {"object", "item", "thing"}
GENERIC_PLURAL_NOUNS = {"objects", "items", "things"}
GENERIC_LOCATION = {"place", "location", "spot", "position", "area", "zone"}
PRECISION_ADVERBS = {
    "exactly",
    "precisely",
    "about",
    "around",
    "approximately",
    "roughly",
}
ORDER_ALTERNATIVES = {"order", "way", "manner", "sequence"}
SCENE_WORDS = {
    "scene",
    "arrangement",
    "configuration",
    "layout",
    "setup",
    "composition",
}
RESTORE_WORDS = {"restore", "put it back", "undo", "reverse", "revert back"}
EXCEEDING_CONSTRAINTS = {
    "exceeding",
    "going over",
    "surpassing",
    "overstepping",
    "going beyond",
    "fully crossing",
}
TOUCHING_CONSTRAINTS = {"touching", "reaching"}
CONTAINER_NOUN = {"container", *GENERIC_SINGULAR_NOUNS, *GENERIC_LOCATION}

SAME_ALTERNATIVES = {"same", "identical", "matching", "similar"}
TEXTURE_ALTERNATIVES = {"texture", "pattern", "look", "style"}
PROFILE_ALTERNATIVES = {"profile", "shape", "form", "structure"}
STARTING_ALTERNATIVES = {"starting", "original", "initial"}
FINALLY_ALTERNATIVES = {"finally", "last", "then"}
MOTION_ALTERNATIVES = {"motion", "movement", "maneuver"}
BUILD_ALTERNATIVES = {"build", "create", "construct", "form", "make"}


def _consecutive_subsets(iterable: list[str]) -> list[str]:
    """Get all consecutive subsets of an iterable."""
    subsets: list[tuple[str, ...]] = [
        beginning
        for beginning, _, _ in more_itertools.windowed_complete(iterable, 1)  # pyright: ignore[reportAssignmentType]
    ]
    # Add the full list onto the end
    subsets.append(tuple(iterable))
    # Remove the empty starting list with none in
    _ = subsets.pop(0)
    return ["".join(subset) for subset in subsets]


class RewordPromptTransform(VIMAInstanceTransform):
    """Overhaul the prompt of a VIMA instance."""

    def __init__(self, *, allow_original_reuse: bool = True, max_attempts: int = 100) -> None:
        self.allow_original_reuse = allow_original_reuse
        self._max_attempts = max_attempts

        self.formatter = TemplateFormatter()

        self.replacers = {
            Task.visual_manipulation: TemplateReplacer(
                task=Task.visual_manipulation,
                original_reuse_allowed=self.allow_original_reuse,
                templates=[
                    "{placing_verb} {article1} {dragged_obj} {preposition} {base_obj}",
                    "{placing_verb} {dragged_obj} {preposition} {base_obj}",
                    "{placing_verb} {dragged_obj} {preposition} {article1} {base_obj}",
                    "{placing_verb} {article1} {dragged_obj} {preposition} {article2} {base_obj}",
                    "{lifting_verb} {dragged_obj} and {placing_verb} {preposition} {base_obj}",
                ],
                key_replacements={
                    "lifting_verb": LIFTING_VERBS,
                    "placing_verb": PLACING_VERBS,
                    "article1": ARTICLE,
                    "article2": ARTICLE,
                    "preposition": PREPOSITIONS,
                },
            ),
            Task.scene_understanding: TemplateReplacer(
                task=Task.scene_understanding,
                original_reuse_allowed=self.allow_original_reuse,
                templates=[
                    "{placing_verb} {article1} {dragged_texture} {singular_noun1} in {scene} {preposition} {article2} {base_texture} {singular_noun2}",
                    "From the {scene} {placing_verb} {article1} {dragged_texture} {singular_noun1} {preposition} {article2} {base_texture} {singular_noun2}",
                    "{placing_verb} {article1} {dragged_texture} {singular_noun1} {preposition} {article2} {base_texture} {singular_noun2} in the {scene}",
                    "{doing_verb} {article2} {base_texture} {singular_noun1} {holding_verb} {article1} {dragged_texture} {singular_noun2} in {scene}",
                    "{moving_verb} {plural_noun1} in the {scene} so that {article1} {dragged_texture} {singular_noun2} are {preposition} {article2} {base_texture} {singular_noun2}",
                ],
                key_replacements={
                    "placing_verb": PLACING_VERBS,
                    "article1": ARTICLE,
                    "article2": ARTICLE,
                    "preposition": PREPOSITIONS,
                    "singular_noun1": GENERIC_SINGULAR_NOUNS,
                    "singular_noun2": GENERIC_SINGULAR_NOUNS,
                    "plural_noun1": GENERIC_PLURAL_NOUNS,
                    "doing_verb": DOING_VERB,
                    "holding_verb": HOLDING_VERBS,
                    "moving_verb": MOVING_VERBS,
                },
            ),
            Task.rotate: TemplateReplacer(
                task=Task.rotate,
                original_reuse_allowed=self.allow_original_reuse,
                templates=[
                    "{rotate_verb} {article1} {dragged_obj} {angle_in_degree} {angle_unit}",
                    "{precision_adverb1} {rotate_verb} {article1} {dragged_obj} {angle_in_degree} {angle_unit}",
                    "{rotate_verb} {article1} {dragged_obj} {precision_adverb1} {angle_in_degree} {angle_unit}",
                    "{pickup_verb} {article1} {dragged_obj} and {rotate_verb} {angle_in_degree} {angle_unit}",
                    "{pickup_verb} {article1} {dragged_obj} and {rotate_verb} it {angle_in_degree} {angle_unit}",
                ],
                key_replacements={
                    "pickup_verb": LIFTING_VERBS,
                    "rotate_verb": ROTATING_VERBS,
                    "article1": ARTICLE,
                    "precision_adverb1": PRECISION_ADVERBS,
                    "angle_unit": ANGLE_UNITS,
                },
            ),
            Task.rearrange: TemplateReplacer(
                task=Task.rearrange,
                original_reuse_allowed=self.allow_original_reuse,
                templates=[
                    "{rearrange_verb} {preposition} this {scene}",
                    "{rearrange_verb} {plural_noun} {preposition} this {scene_word} {scene}",
                ],
                key_replacements={
                    "rearrange_verb": REARRANGE_VERBS,
                    "preposition": REARRANGE_PREPOSITIONS,
                    "plural_noun": GENERIC_PLURAL_NOUNS,
                    "scene_word": SCENE_WORDS,
                },
            ),
            Task.rearrange_then_restore: TemplateReplacer(
                task=Task.rearrange_then_restore,
                original_reuse_allowed=self.allow_original_reuse,
                templates=[
                    "{rearrange_verb} {preposition} this {scene} and then {restore_word}",
                    "{rearrange_verb} {preposition} this {scene} and {restore_word}",
                    "{rearrange_verb} {plural_noun} {preposition} this {scene_word} {scene} and then {restore_word}",
                    "{rearrange_verb} {plural_noun} {preposition} this {scene_word} {scene} and {restore_word}",
                ],
                key_replacements={
                    "rearrange_verb": REARRANGE_VERBS,
                    "preposition": REARRANGE_PREPOSITIONS,
                    "plural_noun": GENERIC_PLURAL_NOUNS,
                    "scene_word": SCENE_WORDS,
                    "restore_word": RESTORE_WORDS,
                },
            ),
            Task.novel_adj: TemplateReplacer(
                task=Task.novel_adj,
                original_reuse_allowed=self.allow_original_reuse,
                templates=[
                    "{demo_blicker_obj_1} is {novel_adj} than {demo_blicker_obj_2}. {demo_blicker_obj_3} is {novel_adj} than {demo_blicker_obj_4}. {demo_blicker_obj_5} is {novel_adj} than {demo_blicker_obj_6}. {placing_verb} the {novel_adj} {dragged_obj} {contain_preposition} the {base_obj}.",
                    "{demo_blicker_obj_1} is {novel_adj} than {demo_blicker_obj_2}. {demo_blicker_obj_3} is {novel_adj} than {demo_blicker_obj_4}. {placing_verb} the {novel_adj} {dragged_obj} {contain_preposition} the {base_obj}.",
                    "{demo_blicker_obj_1} is {novel_adj} than {demo_blicker_obj_2}. {placing_verb} the {novel_adj} {dragged_obj} {contain_preposition} the {base_obj}.",
                    "{object1} {object3} and {object5} are all {novel_adj} than {plural_noun} {object2} {object4} and {object6} respectively. {placing_verb} the {novel_adj} {dragged_obj} {contain_preposition} the {base_obj}",
                    "{object1} and {object3} are both {novel_adj} than {plural_noun} {object2} and {object4} respectively. {placing_verb} the {novel_adj} {dragged_obj} {contain_preposition} the {base_obj}",
                    "{demo_blicker_obj_1} is {novel_adj} than {demo_blicker_obj_2}. {demo_blicker_obj_3} is {novel_adj} than {demo_blicker_obj_4}. {demo_blicker_obj_5} is {novel_adj} than {demo_blicker_obj_6}. {placing_verb} the {adv} {novel_adj} {dragged_obj} {contain_preposition} the {base_obj}.",
                    "{demo_blicker_obj_1} is {novel_adj} than {demo_blicker_obj_2}. {demo_blicker_obj_3} is {novel_adj} than {demo_blicker_obj_4}. {placing_verb} the {adv} {novel_adj} {dragged_obj} {contain_preposition} the {base_obj}.",
                    "{demo_blicker_obj_1} is {novel_adj} than {demo_blicker_obj_2}. {placing_verb} the {adv} {novel_adj} {dragged_obj} {contain_preposition} the {base_obj}.",
                    "{object1} {object3} and {object5} are all {novel_adj} than {plural_noun} {object2} {object4} and {object6} respectively. {placing_verb} the {adv} {novel_adj} {dragged_obj} {contain_preposition} the {base_obj}",
                    "{object1} and {object3} are both {novel_adj} than {plural_noun} {object2} and {object4} respectively. {placing_verb} the {adv} {novel_adj} {dragged_obj} {contain_preposition} the {base_obj}",
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
            Task.novel_noun: TemplateReplacer(
                task=Task.novel_noun,
                original_reuse_allowed=self.allow_original_reuse,
                templates=[
                    "This is a {novel_noun1} {object1}. This is a {novel_noun2} {object2}. {placing_verb} the {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    "This is a {novel_noun1} {object1} and this is a {novel_noun2} {object2}. {placing_verb} the {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    "This is a {novel_noun2} {object2}. This is a {novel_noun1} {object1}. {placing_verb} the {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    "This is a {novel_noun2} {object2} and this is a {novel_noun1} {object1}. {placing_verb} the {novel_noun1} {contain_preposition} the {novel_noun2}.",
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
            Task.novel_adj_and_noun: TemplateReplacer(
                task=Task.novel_adj_and_noun,
                original_reuse_allowed=self.allow_original_reuse,
                templates=[
                    "{this} is a {novel_noun1} {object1}. {this} is a {novel_noun2} {object2}. {object3} is {novel_adj} than {object4}. {object5} is {novel_adj} than {object6}. {object7} is {novel_adj} than {object8}. {placing_verb} the {adv} {novel_adj} {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    "{this} is a {novel_noun1} {object1}. {this} is a {novel_noun2} {object2}. {object3} {object5} and {object7} are all {novel_adj} than {plural_noun} {object4} {object6} and {object8} respectively. {placing_verb} the {adv} {novel_adj} {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    "{this} is a {novel_noun1} {object1}. {this} is a {novel_noun2} {object2}. {object3} is {novel_adj} than {object4}. {object5} is {novel_adj} than {object6}. {object7} is {novel_adj} than {object8}. {placing_verb} the {adv} {novel_adj} {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    "{this} is a {novel_noun2} {object2}. {this} is a {novel_noun1} {object1}. {object3} is {novel_adj} than {object4}. {object5} is {novel_adj} than {object6}. {object7} is {novel_adj} than {object8}. {placing_verb} the {adv} {novel_adj} {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    "{object3} is {novel_adj} than {object4}. {object5} is {novel_adj} than {object6}. {object7} is {novel_adj} than {object8}. {this} is a {novel_noun2} {object2}. {this} is a {novel_noun1} {object1}. {placing_verb} the {adv} {novel_adj} {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    "{object3} {object5} and {object7} are all {novel_adj} than {plural_noun} {object4} {object6} and {object8} respectively. {this} is a {novel_noun2} {object2}. {this} is a {novel_noun1} {object1}. {placing_verb} the {adv} {novel_adj} {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    # Same as above but with 2 adj compares
                    "{this} is a {novel_noun1} {object1}. {this} is a {novel_noun2} {object2}. {object3} is {novel_adj} than {object4}. {object5} is {novel_adj} than {object6}. {placing_verb} the {adv} {novel_adj} {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    "{this} is a {novel_noun2} {object2}. {this} is a {novel_noun1} {object1}. {object3} is {novel_adj} than {object4}. {object5} is {novel_adj} than {object6}. {placing_verb} the {adv} {novel_adj} {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    "{object3} is {novel_adj} than {object4}. {object5} is {novel_adj} than {object6}. {this} is a {novel_noun2} {object2}. {this} is a {novel_noun1} {object1}. {placing_verb} the {adv} {novel_adj} {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    # Same as above, but with one less adj comparison
                    "{this} is a {novel_noun1} {object1}. {this} is a {novel_noun2} {object2}. {object3} is {novel_adj} than {object4}. {placing_verb} the {adv} {novel_adj} {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    "{this} is a {novel_noun2} {object2}. {this} is a {novel_noun1} {object1}. {object3} is {novel_adj} than {object4}. {placing_verb} the {adv} {novel_adj} {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    "{object3} is {novel_adj} than {object4}. {this} is a {novel_noun2} {object2}. {this} is a {novel_noun1} {object1}. {placing_verb} the {adv} {novel_adj} {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    # and without {adv}
                    "{this} is a {novel_noun1} {object1}. {this} is a {novel_noun2} {object2}. {object3} is {novel_adj} than {object4}. {object5} is {novel_adj} than {object6}. {object7} is {novel_adj} than {object8}. {placing_verb} the {novel_adj} {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    "{this} is a {novel_noun1} {object1}. {this} is a {novel_noun2} {object2}. {object3} {object5} and {object7} are all {novel_adj} than {plural_noun} {object4} {object6} and {object8} respectively. {placing_verb} the {novel_adj} {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    "{this} is a {novel_noun1} {object1}. {this} is a {novel_noun2} {object2}. {object3} is {novel_adj} than {object4}. {object5} is {novel_adj} than {object6}. {object7} is {novel_adj} than {object8}. {placing_verb} the {novel_adj} {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    "{this} is a {novel_noun2} {object2}. {this} is a {novel_noun1} {object1}. {object3} is {novel_adj} than {object4}. {object5} is {novel_adj} than {object6}. {object7} is {novel_adj} than {object8}. {placing_verb} the {novel_adj} {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    "{object3} is {novel_adj} than {object4}. {object5} is {novel_adj} than {object6}. {object7} is {novel_adj} than {object8}. {this} is a {novel_noun2} {object2}. {this} is a {novel_noun1} {object1}. {placing_verb} the {novel_adj} {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    "{object3} {object5} and {object7} are all {novel_adj} than {plural_noun} {object4} {object6} and {object8} respectively. {this} is a {novel_noun2} {object2}. {this} is a {novel_noun1} {object1}. {placing_verb} the {novel_adj} {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    # Same as above but with 2 adj compares
                    "{this} is a {novel_noun1} {object1}. {this} is a {novel_noun2} {object2}. {object3} is {novel_adj} than {object4}. {object5} is {novel_adj} than {object6}. {placing_verb} the {novel_adj} {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    "{this} is a {novel_noun2} {object2}. {this} is a {novel_noun1} {object1}. {object3} is {novel_adj} than {object4}. {object5} is {novel_adj} than {object6}. {placing_verb} the {novel_adj} {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    "{object3} is {novel_adj} than {object4}. {object5} is {novel_adj} than {object6}. {this} is a {novel_noun2} {object2}. {this} is a {novel_noun1} {object1}. {placing_verb} the {novel_adj} {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    # Same as above, but with one less adj comparison
                    "{this} is a {novel_noun1} {object1}. {this} is a {novel_noun2} {object2}. {object3} is {novel_adj} than {object4}. {placing_verb} the {novel_adj} {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    "{this} is a {novel_noun2} {object2}. {this} is a {novel_noun1} {object1}. {object3} is {novel_adj} than {object4}. {placing_verb} the {novel_adj} {novel_noun1} {contain_preposition} the {novel_noun2}.",
                    "{object3} is {novel_adj} than {object4}. {this} is a {novel_noun2} {object2}. {this} is a {novel_noun1} {object1}. {placing_verb} the {novel_adj} {novel_noun1} {contain_preposition} the {novel_noun2}.",
                ],
                key_replacements={
                    "rearrange_verb": REARRANGE_VERBS,
                    "preposition": REARRANGE_PREPOSITIONS,
                    "plural_noun": GENERIC_PLURAL_NOUNS,
                    "scene_word": SCENE_WORDS,
                    "restore_word": RESTORE_WORDS,
                    "placing_verb": PLACING_VERBS,
                    "contain_preposition": CONTAIN_PREPOSITION,
                    "this": {"this"},
                },
            ),
            Task.twist: TemplateReplacer(
                task=Task.twist,
                original_reuse_allowed=self.allow_original_reuse,
                templates=[
                    '"Twist" is defined as {rotating_word} some {singular_noun} a specific {angle_word}. For examples: From {before_twist1} to {after_twist1}. From {before_twist2} to {after_twist2}. Now twist all {texture} objects.',
                    '"Twist" is defined as {rotating_word} some {singular_noun} a specific {angle_word}. For examples: From {before_twist1} to {after_twist1}. From {before_twist2} to {after_twist2}. From {before_twist3} to {after_twist3}. Now twist all {texture} objects.',
                    '"Twist" is defined as {rotating_word} some {singular_noun} a specific {angle_word}. For examples: From {before_twist1} to {after_twist1}. Now twist all {texture} objects.',
                    '"Twist" is defined as {rotating_word} {singular_noun} a specific {angle_word}. For examples: From {before_twist1} to {after_twist1}. From {before_twist2} to {after_twist2}. Now twist all {texture} objects.',
                    '"Twist" is defined as {rotating_word} {singular_noun} a specific {angle_word}. For examples: From {before_twist1} to {after_twist1}. From {before_twist2} to {after_twist2}. From {before_twist3} to {after_twist3}. Now twist all {texture} objects.',
                    '"Twist" is defined as {rotating_word} {singular_noun} a specific {angle_word}. For examples: From {before_twist1} to {after_twist1}. Now twist all {texture} objects.',
                    '"Twist" is defined as {rotating_word} {plural_noun} a specific {angle_word}. For examples: From {before_twist1} to {after_twist1}. From {before_twist2} to {after_twist2}. Now twist all {texture} objects.',
                    '"Twist" is defined as {rotating_word} {plural_noun} a specific {angle_word}. For examples: From {before_twist1} to {after_twist1}. From {before_twist2} to {after_twist2}. From {before_twist3} to {after_twist3}. Now twist all {texture} objects.',
                    '"Twist" is defined as {rotating_word} {plural_noun} a specific {angle_word}. For examples: From {before_twist1} to {after_twist1}. Now twist all {texture} objects.',
                ],
                key_replacements={
                    "rotating_word": ROTATING_ALTERNATIVES,
                    "plural_noun": GENERIC_PLURAL_NOUNS,
                    "singular_noun": GENERIC_SINGULAR_NOUNS,
                    "angle_word": {"angle", "way", "amount"},
                },
            ),
            Task.follow_motion: TemplateReplacer(
                task=Task.follow_motion,
                original_reuse_allowed=self.allow_original_reuse,
                templates=[
                    *[
                        "{moving_verb} the {object1} like this:" + subset
                        for subset in _consecutive_subsets(
                            [f" {{frame{idx}}}" for idx in range(1, 4)]
                        )
                    ],
                    *[
                        "{moving_verb} the {object1} in this {way_word}:" + subset
                        for subset in _consecutive_subsets(
                            [f" {{frame{idx}}}" for idx in range(1, 4)]
                        )
                    ],
                    # *[
                    #     "{follow_verb} this {motion_word}:" + subset
                    #     for subset in _consecutive_subsets(
                    #         [f" {{frame{idx}}}" for idx in range(1, 4)]
                    #     )
                    # ],
                    *[
                        "{follow_verb} this {motion_word} for {object1}:" + subset
                        for subset in _consecutive_subsets(
                            [f" {{frame{idx}}}" for idx in range(1, 4)]
                        )
                    ],
                ],
                key_replacements={
                    "moving_verb": MOVING_VERBS,
                    "follow_verb": FOLLOW_VERBS,
                    "motion_word": MOTION_ALTERNATIVES,
                    "way_word": ORDER_ALTERNATIVES,
                },
            ),
            Task.follow_order: TemplateReplacer(
                task=Task.follow_order,
                original_reuse_allowed=self.allow_original_reuse,
                templates=[
                    *[
                        "{stack_verb} {plural_noun} in this {order_word}:" + subset
                        for subset in _consecutive_subsets(
                            [f" {{frame{idx}}}" for idx in range(1, 4)]
                        )
                    ],
                    *[
                        "{build_noun} a {stack_noun} of {plural_noun} in this {order_word}:"
                        + subset
                        for subset in _consecutive_subsets(
                            [f" {{frame{idx}}}" for idx in range(1, 4)]
                        )
                    ],
                    *[
                        "Follow this sequence to {build_noun} a {stack_noun} of {plural_noun}:"
                        + subset
                        for subset in _consecutive_subsets(
                            [f" {{frame{idx}}}" for idx in range(1, 4)]
                        )
                    ],
                    *[
                        "Follow this sequence to {stack_verb} {plural_noun}:" + subset
                        for subset in _consecutive_subsets(
                            [f" {{frame{idx}}}" for idx in range(1, 4)]
                        )
                    ],
                ],
                key_replacements={
                    "stack_verb": {*STACK_VERBS, "organize"},
                    "stack_noun": {"stack", "pile", "tower", "structure"},
                    "order_word": ORDER_ALTERNATIVES,
                    "plural_noun": GENERIC_PLURAL_NOUNS,
                    "build_noun": BUILD_ALTERNATIVES,
                },
            ),
            Task.sweep_without_exceeding: TemplateReplacer(
                task=Task.sweep_without_exceeding,
                original_reuse_allowed=self.allow_original_reuse,
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
                original_reuse_allowed=self.allow_original_reuse,
                templates=[
                    "{sweep_verb} {quantity} {object1} {preposition} {bounds} without {touching_constraint} {constraint}",
                    "{sweep_verb} {quantity} {object1} {preposition} the {bounds} without {touching_constraint} the {constraint}",
                    "without {touching_constraint} the {constraint}, {sweep_verb} {quantity} {object1} {preposition} the {bounds}",
                ],
                key_replacements={
                    "sweep_verb": SWEEP_VERBS,
                    "preposition": CONTAIN_PREPOSITION,
                    "touching_constraint": TOUCHING_CONSTRAINTS,
                },
            ),
            Task.same_texture: TemplateReplacer(
                task=Task.same_texture,
                original_reuse_allowed=self.allow_original_reuse,
                templates=[
                    "{placing_verb} {quantity} {plural_noun} {association_preposition} {article1} {same} {texture} as {object1} {contain_preposition} it",
                    "{placing_verb} {quantity} {article2} {plural_noun} {association_preposition} {article1} {same} {texture} as {object1} {contain_preposition} it",
                ],
                key_replacements={
                    "placing_verb": PLACING_VERBS,
                    "quantity": PLURAL_PRONOUNS,
                    "plural_noun": GENERIC_PLURAL_NOUNS,
                    "association_preposition": ASSOCIATION_PREPOSITIONS,
                    "contain_preposition": CONTAIN_PREPOSITION,
                    "article1": {"the"},
                    "article2": PREPOSITION_ARTICLE,
                    "texture": TEXTURE_ALTERNATIVES,
                    "same": SAME_ALTERNATIVES,
                },
            ),
            Task.same_shape: TemplateReplacer(
                task=Task.same_shape,
                original_reuse_allowed=self.allow_original_reuse,
                templates=[
                    "{placing_verb} {quantity} {plural_noun} {association_preposition} {article1} {same} {profile} as {object1} {contain_preposition} it",
                    "{placing_verb} {quantity} {article2} {plural_noun} {association_preposition} {article1} {same} {profile} as {object1} {contain_preposition} it",
                ],
                key_replacements={
                    "placing_verb": PLACING_VERBS,
                    "quantity": PLURAL_PRONOUNS,
                    "plural_noun": GENERIC_PLURAL_NOUNS,
                    "association_preposition": ASSOCIATION_PREPOSITIONS,
                    "contain_preposition": CONTAIN_PREPOSITION,
                    "article1": {"the"},
                    "article2": PREPOSITION_ARTICLE,
                    "profile": PROFILE_ALTERNATIVES,
                    "same": SAME_ALTERNATIVES,
                },
            ),
            Task.manipulate_old_neighbor: TemplateReplacer(
                task=Task.manipulate_old_neighbor,
                original_reuse_allowed=self.allow_original_reuse,
                templates=[
                    "{preceeding_adjective} {placing_verb} {object1} {contain_preposition} {object2} then {placing_verb2} {article1} {singular_noun} that was previously at its {direction} into the same {object2}",
                    "{placing_verb} {object1} {contain_preposition} {object2} then {placing_verb2} {article1} {singular_noun} that was at its {direction} {contain_preposition2} the same {object2}",
                    "{placing_verb} {object1} {contain_preposition} {object2} then {placing_verb2} {article1} {singular_noun} that was at its {direction} {contain_preposition2} {object2}",
                    "{placing_verb} {object1} {contain_preposition} {object2} then {placing_verb2} {article1} {singular_noun} that was at its {direction} before you {moved_verb} it into the same {object2}",
                    "{placing_verb} {object1} {contain_preposition} {object2} then {placing_verb2} {article1} {singular_noun} that was at its {direction} before you {moved_verb} it into the same as {object1}",
                    "{placing_verb} {object1} {contain_preposition} {object2} then {placing_verb2} {article1} {singular_noun} that was at its {direction} before you {moved_verb} it into the same {container_noun}",
                    "{placing_verb} {object1} and {article1} {singular_noun} at its {direction} {contain_preposition} {object2} starting with {object1}",
                    "{placing_verb} {object1} and {article1} {singular_noun} at its {direction} {contain_preposition} {object2} with {object1} first",
                    "{placing_verb} {object1} and {article1} {singular_noun} at its {direction} {contain_preposition} {object2} with {object1} at the bottom",
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
            Task.pick_in_order_then_restore: TemplateReplacer(
                task=Task.pick_in_order_then_restore,
                original_reuse_allowed=self.allow_original_reuse,
                templates=[
                    "{placing_verb} {object1} {contain_preposition} {object2}. {finally_alternative} {restore_word} it {contain_preposition2} {article1} {starting_alternative} {container_noun}",
                    "{placing_verb} {object1} {contain_preposition} {object2} then {object3}. {finally_alternative} {restore_word} it {contain_preposition2} {article1} {starting_alternative} {container_noun}",
                    "{placing_verb} {object1} {contain_preposition} {object2} and then {restore_word}",
                    "{placing_verb} {object1} {contain_preposition} {object2} and {object3} and then {restore_word}",
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
        }

    def __call__(self, instance: VIMAInstance) -> VIMAInstance:
        """Overhaul the prompt of a VIMA instance."""
        original_prompt = instance.prompt
        necessary_placeholders = list(instance.prompt_assets.as_dict.keys())

        # We need to first extract the necessary placeholders from the original prompt
        keys_from_original = self._extract_keys_from_original(instance)

        # Then we need to generate a new prompt
        new_prompt = self.replacers[instance.task].generate_new_prompt(
            original_prompt, keys_from_original, necessary_placeholders
        )

        # And then we return the instance
        return instance.model_copy(deep=True, update={"prompt": new_prompt})

    def generate_all_possible_prompts(self, instance: VIMAInstance) -> set[str]:
        """Generate all possible prompts for a given prompt."""
        necessary_placeholders = list(instance.prompt_assets.as_dict.keys())
        keys_from_original = self._extract_keys_from_original(instance)
        all_generated_templates = self.replacers[instance.task].generate_all_possible_prompts(
            keys_from_original, necessary_placeholders
        )
        return all_generated_templates

    def _extract_keys_from_original(self, instance: VIMAInstance) -> dict[str, str]:
        """Extract keys from the original prompt."""
        prompt = instance.prompt
        # First, check if there are any textures within the prompt itself, and if so, convert them
        # to texture placeholders
        texture_placeholders = {}
        sorted_texture_names = [
            metadata.texture_name
            for metadata in sorted(
                instance.object_metadata,
                key=lambda metadata: len(metadata.texture_name),
                reverse=True,
            )
        ]
        for texture_name in sorted_texture_names:
            if texture_name in prompt:
                texture_counter = len(texture_placeholders)
                texture_placeholders[f"texture_{texture_counter}"] = texture_name
                prompt = prompt.replace(texture_name, f"{{texture_{texture_counter}}}", 1)

        # Then we get the template that matches the prompt
        template = self.replacers[instance.task].get_template_that_best_matches_prompt(prompt)

        keys_from_original = extract_keys_from_original(prompt, template)

        # Put all the textures back
        for key, extracted_value in keys_from_original.items():
            with suppress(KeyError):
                keys_from_original[key] = extracted_value.format_map(texture_placeholders)

        return keys_from_original
