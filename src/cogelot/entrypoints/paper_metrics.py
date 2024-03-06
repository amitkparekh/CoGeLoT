import statistics
from contextlib import suppress
from decimal import Decimal
from typing import Annotated, ClassVar, Literal

import typer
import wandb
from loguru import logger
from rich.console import Console
from wandb.sdk.wandb_run import Run

from cogelot.data.evaluation import VIMAEvaluationDataset
from cogelot.data.transforms import (
    GobbledyGookPromptTokenTransform,
    GobbledyGookPromptWordTransform,
    TextualDescriptionTransform,
)
from cogelot.environment.vima import VIMAEnvironment
from cogelot.modules.tokenizers.text import TextTokenizer
from cogelot.structures.vima import Task

console = Console()

PasteTarget = Literal["paper", "excel", "draft"]
Level = Literal["L1", "L2", "L3", "L4"]


class EvaluationPerformancePrinter:
    """Print the evaluation performance in a format that can be easily pasted."""

    delimiter_per_target: ClassVar[dict[PasteTarget, str]] = {
        "paper": " & ",
        "draft": " & ",
        "excel": ",",
    }
    success_prefix: ClassVar[str] = "success/"
    levels: ClassVar[list[Level]] = ["L1", "L2", "L3", "L4"]
    textual_override: ClassVar[str] = "{---}"

    def __init__(self, *, target: PasteTarget) -> None:
        self.target = target
        self.delimiter = self.delimiter_per_target[target]

    def __call__(self, run_id: str) -> None:
        """Print the performance in a format that can be easily pasted."""
        run = self.get_run(run_id)
        is_textual = "textual" in run.name.lower()
        performances = self.get_evaluation_performance(run)

        if self.target == "paper":
            self.print_for_paper(performances, is_textual=is_textual)
        if self.target == "excel":
            self.print_for_excel(performances, is_textual=is_textual)
        if self.target == "draft":
            self.print_for_paper_draft(performances)

    def get_run(self, run_id: str) -> Run:
        """Get the run from WandB."""
        run = wandb.Api().run(f"pyop/cogelot-evaluation/{run_id}")
        # Print information about the run
        console.print("Run:", run.id)
        console.print("Name:", run.name)
        with suppress(KeyError):
            console.print("Training dataset:", run.config["training_data"])

        return run

    def get_evaluation_performance(self, run: Run) -> dict[Level, dict[int, Decimal]]:
        """Get the evaluation performance from WandB."""
        performance_per_level = {}
        # Get all the success metrics per partition
        for level in self.levels:
            task_success = {
                int(key[-2:]): Decimal(success * 100).quantize(Decimal("1.0"))  # noqa: WPS221
                for key, success in sorted(run.summary.items())  # pyright: ignore[reportCallIssue]
                if key.startswith(f"{self.success_prefix}{level}")
            }
            performance_per_level[level] = task_success

        return performance_per_level

    def print_for_paper(
        self, performances: dict[Level, dict[int, Decimal]], *, is_textual: bool
    ) -> None:
        """Print the performance in a format that can be easily pasted in LaTeX."""
        for level, task_success in performances.items():
            if is_textual:
                task_success = {  # noqa: PLW2901
                    task_num: task_value
                    if Task(task_num - 1) not in TextualDescriptionTransform.tasks_to_avoid
                    else "{---}"
                    for task_num, task_value in task_success.items()
                }
            average = statistics.mean(
                task_value
                for task_value in task_success.values()
                if isinstance(task_value, Decimal)
            )
            print_line = self.delimiter.join(map(str, task_success.values()))
            print_line += self.delimiter + str(average.quantize(Decimal("1.0")))
            console.print(f"{level} Success")
            console.print(print_line)

    def print_for_paper_draft(self, performances: dict[Level, dict[int, Decimal]]) -> None:
        """Print the performance in a format that can be easily pasted in LaTeX."""
        for level, task_success in performances.items():
            task_success = {  # noqa: PLW2901
                task.value + 1: task_success.get(task.value + 1, r"{{{---}}}") for task in Task
            }
            average = statistics.mean(
                task_value
                for task_value in task_success.values()
                if isinstance(task_value, Decimal)
            )
            print_line = self.delimiter.join(map(str, task_success.values()))
            print_line += self.delimiter + str(average.quantize(Decimal("1.0")))
            console.print(f"{level} &")
            console.print(print_line)

    def print_for_excel(
        self, performances: dict[Level, dict[int, Decimal]], *, is_textual: bool
    ) -> None:
        """Print the performance in a format that can be easily pasted in Excel."""
        print_line = ""
        for task_success in performances.values():
            if is_textual:
                task_success = {  # noqa: PLW2901
                    task_num: task_value
                    if Task(task_num - 1) not in TextualDescriptionTransform.tasks_to_avoid
                    else ""
                    for task_num, task_value in task_success.items()
                }

            if print_line:
                print_line += self.delimiter
            print_line += self.delimiter.join(map(str, task_success.values()))

        console.print(print_line)


def print_performance(run_id: str, *, target: Annotated[str, typer.Option()]) -> None:
    """Get the evaluation performance from WandB for easy pasting."""
    assert target in EvaluationPerformancePrinter.delimiter_per_target
    printer = EvaluationPerformancePrinter(target=target)
    printer(run_id)


def print_prompt_lengths() -> None:
    """Get the lengths of different types of prompts."""
    text_tokenizer = TextTokenizer("t5-base")
    evaluation_dataset = VIMAEvaluationDataset.from_partition_to_specs(num_repeats_per_episode=1)
    environment = VIMAEnvironment.from_config(task=1, partition=1, seed=0)
    gobbledygook_word_transform = GobbledyGookPromptWordTransform()
    gobbledygook_token_transform = GobbledyGookPromptTokenTransform(text_tokenizer, timeout=100)

    original_instructions: list[tuple[str, list[int]]] = []
    gobbledygook_words: list[tuple[str, list[int]]] = []
    gobbledygook_tokens: list[tuple[str, list[int]]] = []
    for partition, task in evaluation_dataset:
        logger.info(f"Partition: {partition}, Task: {task}")
        environment.set_task(task, partition)
        environment.reset()
        # environment.render()

        logger.info("Creating VIMA instance")
        instance = environment.create_vima_instance()
        original_instructions.append(
            (instance.prompt, text_tokenizer.tokenizer.encode(instance.prompt))
        )

        logger.info("Transforming the instance (GDG Words)")
        gdg_word_instance = gobbledygook_word_transform(instance)
        gobbledygook_words.append(
            (
                gdg_word_instance.prompt,
                text_tokenizer.tokenizer.encode(gdg_word_instance.prompt),
            )
        )

        logger.info("Transforming the instance (GDG Tokens)")
        gdg_tokens_instance = gobbledygook_token_transform(instance)
        gobbledygook_tokens.append(
            (
                gdg_tokens_instance.prompt,
                text_tokenizer.tokenizer.encode(gdg_tokens_instance.prompt),
            )
        )

    console.print("Original Instructions")
    original_instruction_words = [len(prompt[0].split(" ")) for prompt in original_instructions]
    original_instruction_tokens = [len(prompt[1]) for prompt in original_instructions]

    console.print("Avg Words:", statistics.mean(original_instruction_words))
    console.print("Std Dev Words:", statistics.stdev(original_instruction_words))

    console.print("Avg Tokens:", statistics.mean(original_instruction_tokens))
    console.print("Std Dev Tokens:", statistics.stdev(original_instruction_tokens))

    console.print("Gobbledygook Words")
    gobbledygook_words_words = [len(prompt[0].split(" ")) for prompt in gobbledygook_words]
    gobbledygook_words_tokens = [len(prompt[1]) for prompt in gobbledygook_words]

    console.print("Avg Words:", statistics.mean(gobbledygook_words_words))
    console.print("Std Dev Words:", statistics.stdev(gobbledygook_words_words))

    console.print("Avg Tokens:", statistics.mean(gobbledygook_words_tokens))
    console.print("Std Dev Tokens:", statistics.stdev(gobbledygook_words_tokens))

    console.print("Gobbledygook Tokens")
    gobbledygook_tokens_words = [len(prompt[0].split(" ")) for prompt in gobbledygook_tokens]
    gobbledygook_tokens_tokens = [len(prompt[1]) for prompt in gobbledygook_tokens]

    console.print("Avg Words:", statistics.mean(gobbledygook_tokens_words))
    console.print("Std Dev Words:", statistics.stdev(gobbledygook_tokens_words))

    console.print("Avg Tokens:", statistics.mean(gobbledygook_tokens_tokens))
    console.print("Std Dev Tokens:", statistics.stdev(gobbledygook_tokens_tokens))


if __name__ == "__main__":
    print_prompt_lengths()
