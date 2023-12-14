import itertools

from rich import print

TASK1_TEMPLATES = {
    "{lifting_verb} {article1} {object1} {preposition} {article2} {object2}",
    "{placing_verb} {article1} {object1} {preposition} {object2}",
    "{placing_verb} {object1} {preposition} {object2}",
    "{placing_verb} {object1} {preposition} {article1} {object2}",
    "{placing_verb} {article1} {object1} {preposition} {article2} {object2}",
    "{lifting_verb} {object1} and {placing_verb} {preposition} {object2}",
    # "find and {placing_verb} {object1} {preposition} {object2}",
    # "find and {placing_verb} {object1} {preposition} {article} {object2}",
    # "find and {placing_verb} {article} {object1} {preposition} {object2}",
    # "find and {placing_verb} {article} {object1} {preposition} {article} {object2}",
}

TASK2_TEMPLATES = {
    "{verb1} {article1} {texture1} {plural_noun} in {scene} {preposition} {article2} {texture2} {singular_noun}"
}

LIFTING_VERBS = {"pick up", "take", "lift", "grab"}
PLACING_VERBS = {"put", "place", "move", "position", "drop", "store"}
ARTICLE = {"the", "one"}
PLURAL_PRONOUNS = {"all", "all the", "the"}
PREPOSITIONS = {
    "onto",
    "into",
    "on top of",
    "within the confines of",
    "on",
}
GENERIC_SINGULAR_NOUNS = {"object", "item", "thing"}
GENERIC_PLURAL_NOUNS = {"objects", "items", "things"}


def create_prompts_for_task1(*, templates: set[str] = TASK1_TEMPLATES) -> set[str]:
    """Create loads of combinations of prompts for Task 1."""
    all_combinations = list(
        itertools.product(templates, LIFTING_VERBS, PLACING_VERBS, ARTICLE, ARTICLE, PREPOSITIONS)
    )
    all_prompts = []
    for template, lifting_verb, placing_verb, article1, article2, preposition in all_combinations:
        all_prompts.append(
            template.format(
                lifting_verb=lifting_verb,
                placing_verb=placing_verb,
                article1=article1,
                article2=article2,
                object1="{object1}",
                object2="{object2}",
                preposition=preposition,
            )
        )
    return set(all_prompts)


def create_prompts_for_task2(*, templates: set[str] = TASK2_TEMPLATES) -> set[str]:
    """Create loads of combinations for Task 2."""
    all_combinations = list(
        itertools.product(
            templates,
            PLACING_VERBS,
            PLURAL_PRONOUNS,
            ARTICLE,
            PREPOSITIONS,
            GENERIC_SINGULAR_NOUNS,
            GENERIC_PLURAL_NOUNS,
        )
    )
    all_prompts = []
    for (
        template,
        placing_verb,
        plural_pronoun,
        article2,
        preposition,
        singular_noun,
        plural_noun,
    ) in all_combinations:
        all_prompts.append(
            template.format(
                verb1=placing_verb,
                article1=plural_pronoun,
                texture1="{texture1}",
                scene="{scene}",
                preposition=preposition,
                article2=article2,
                texture2="{texture2}",
                singular_noun=singular_noun,
                plural_noun=plural_noun,
            )
        )
    return set(all_prompts)


if __name__ == "__main__":
    task1_prompts = create_prompts_for_task1()
    task2_prompts = create_prompts_for_task2()

    print(task2_prompts)
    print("Task1", len(task1_prompts))
    print("Task2", len(task2_prompts))
