from pytest_cases import fixture, param_fixtures

from cogelot.data.templating.replacers import PerTaskTemplateReplacer
from cogelot.entrypoints.dump_all_templates import ORIGINAL_PROMPTS
from cogelot.structures.vima import Task


@fixture(scope="module")
def per_task_template_replacer() -> PerTaskTemplateReplacer:
    return PerTaskTemplateReplacer(allow_original_reuse=False)


task, original = param_fixtures(  # pyright: ignore[reportGeneralTypeIssues]
    "task, original", ORIGINAL_PROMPTS
)


def test_prompt_regenerated_successfully(
    task: Task, original: str, per_task_template_replacer: PerTaskTemplateReplacer
) -> None:
    new_prompt = per_task_template_replacer.regenerate_instruction(task, original)
    assert original != new_prompt
