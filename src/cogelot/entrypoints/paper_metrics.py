import statistics
from contextlib import suppress
from decimal import Decimal

import wandb
from loguru import logger
from rich.console import Console

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


def print_performance(run_id: str) -> None:
    """Get the evaluation performance from WandB for easy pasting in the paper."""
    # Load the run
    run = wandb.Api().run(f"pyop/cogelot-evaluation/{run_id}")

    # Print information about the run
    console.print("Run:", run.id)
    console.print("Name:", run.name)
    with suppress(KeyError):
        console.print("Training dataset:", run.config["training_data"])

    is_textual = "textual" in run.name.lower()

    # Get all the success metrics per partition
    for level in ("L1", "L2", "L3", "L4"):
        task_success = {
            int(key[-2:]): Decimal(success * 100).quantize(Decimal("1.0"))  # noqa: WPS221
            for key, success in sorted(run.summary.items())
            if key.startswith(f"success/{level}")
        }

        if is_textual:
            task_success = {
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

        console.print(f"{level} Success")
        console.print(
            " & ".join(map(str, task_success.values()))
            + " & "
            + str(average.quantize(Decimal("1.0")))
        )


def print_prompt_lengths() -> None:
    """Get the lengths of different types of prompts."""
    text_tokenizer = TextTokenizer("t5-base")
    evaluation_dataset = VIMAEvaluationDataset.from_partition_to_specs(
        num_repeats_per_episode=1
    )
    environment = VIMAEnvironment.from_config(task=1, partition=1, seed=0)
    gobbledygook_word_transform = GobbledyGookPromptWordTransform()
    gobbledygook_token_transform = GobbledyGookPromptTokenTransform(
        text_tokenizer, timeout=100
    )

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
    original_instruction_words = [
        len(prompt[0].split(" ")) for prompt in original_instructions
    ]
    original_instruction_tokens = [len(prompt[1]) for prompt in original_instructions]

    console.print("Avg Words:", statistics.mean(original_instruction_words))
    console.print("Std Dev Words:", statistics.stdev(original_instruction_words))

    console.print("Avg Tokens:", statistics.mean(original_instruction_tokens))
    console.print("Std Dev Tokens:", statistics.stdev(original_instruction_tokens))

    console.print("Gobbledygook Words")
    gobbledygook_words_words = [
        len(prompt[0].split(" ")) for prompt in gobbledygook_words
    ]
    gobbledygook_words_tokens = [len(prompt[1]) for prompt in gobbledygook_words]

    console.print("Avg Words:", statistics.mean(gobbledygook_words_words))
    console.print("Std Dev Words:", statistics.stdev(gobbledygook_words_words))

    console.print("Avg Tokens:", statistics.mean(gobbledygook_words_tokens))
    console.print("Std Dev Tokens:", statistics.stdev(gobbledygook_words_tokens))

    console.print("Gobbledygook Tokens")
    gobbledygook_tokens_words = [
        len(prompt[0].split(" ")) for prompt in gobbledygook_tokens
    ]
    gobbledygook_tokens_tokens = [len(prompt[1]) for prompt in gobbledygook_tokens]

    console.print("Avg Words:", statistics.mean(gobbledygook_tokens_words))
    console.print("Std Dev Words:", statistics.stdev(gobbledygook_tokens_words))

    console.print("Avg Tokens:", statistics.mean(gobbledygook_tokens_tokens))
    console.print("Std Dev Tokens:", statistics.stdev(gobbledygook_tokens_tokens))


if __name__ == "__main__":
    print_prompt_lengths()
