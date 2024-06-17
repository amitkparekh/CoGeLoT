import statistics
from contextlib import suppress
from decimal import Decimal
from pathlib import Path
from typing import Annotated, ClassVar, Literal

import orjson
import polars as pl
import typer
import wandb
from loguru import logger
from rich.console import Console
from wandb.apis.public.runs import Run

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

PasteTarget = Literal["paper", "short", "excel", "full"]
Level = Literal["L1", "L2", "L3", "L4"]

BaselineRuns = {
    "X+Obj / Orig": "sm0t3gea",
    "X+Ptch / Orig": "q95fzf7i",
    "D+Obj / Orig": "l2whu45i",
    "D+Ptch / Orig": "x9k1z1hu",
    "X+Obj / Para": "wsqkbn9g",
    "X+Ptch / Para": "4x7w18dx",
    "D+Obj / Para": "1thybzlh",
    "D+Ptch / Para": "bkrjv2wv",
}


def get_latex_model_name(run: Run) -> str:
    """Get the model name from the run name."""
    prompt_name = {"xattn": "Cross-Attn", "dec_only": "Concatenate"}
    visual_name = {
        "obj_centric": "Obj-Centric",
        "patches": "Patches",
    }
    prompt_style = run.config.get("prompt_conditioning_style")
    visual_style = run.config.get("visual_encoder_style")
    return f"{prompt_name[prompt_style]} + {visual_name[visual_style]}"


def get_baseline_run_id(run: Run) -> str:
    """Get the baseline run id for the given run."""
    for run_name, run_id in BaselineRuns.items():
        before, after = run_name.split(" / ")
        if run.name.startswith(before) and f" / {after}" in run.name:
            return run_id
    raise ValueError(f"Run {run.id} with name {run.name} does not have a baseline.")


def get_run(run_id: str) -> Run:
    """Get the run from WandB."""
    run = wandb.Api().run(f"pyop/cogelot-evaluation/{run_id}")
    # Print information about the run
    console.print("Run:", run.id)
    console.print("Name:", run.name)
    with suppress(KeyError):
        console.print("Training dataset:", run.config["training_data"])
    return run


def download_episodes_from_run(run: Run) -> str:
    """Download the episodes from the given run id."""
    table = next(run.logged_artifacts())
    assert "episodes" in table.name
    table_path = table.download(root=f"./storage/artifacts/{run.id}") + "/episodes.table.json"
    return table_path


def compute_per_task_performance(episodes_df: pl.DataFrame) -> pl.DataFrame:
    """Compute per-task performance from episodes."""
    return (
        episodes_df.group_by(["partition", "task"])
        .agg(
            pl.col("is_successful_at_end").sum().alias("num_successful"),
            pl.col("is_successful_without_mistakes").sum().alias("num_successful_strict"),
        )
        .join(
            episodes_df.group_by(["partition", "task"]).agg(
                pl.col("time_limit").count().alias("total_episodes")
            ),
            on=["partition", "task"],
        )
        .with_columns(
            pl.col("num_successful_strict")
            .truediv(pl.col("total_episodes"))
            .alias("percentage_successful_strict"),
            pl.col("num_successful")
            .truediv(pl.col("total_episodes"))
            .alias("percentage_successful"),
        )
        .sort(["partition", "task"])
    )


def process_episodes_table_from_wandb(path: str) -> pl.DataFrame:
    """Process the episodes table from WandB."""
    raw_episodes = orjson.loads(Path(path).read_text())

    episodes = pl.DataFrame(raw_episodes["data"])
    episodes.columns = raw_episodes["columns"]

    episodes = episodes.select(
        pl.col("partition", "is_successful_at_end"),
        pl.col("minimum_steps").alias("time_limit"),
        pl.col("total_steps").alias("steps_taken"),
        pl.col("task").add(1).alias("task"),
    ).with_columns(
        pl.when(pl.col("is_successful_at_end"))
        .then(pl.col("time_limit").sub(pl.col("steps_taken")).ge(0))
        .otherwise(pl.lit(False))  # noqa: FBT003
        .alias("is_successful_without_mistakes")
    )
    return episodes


class EvaluationPerformancePrinter:
    """Print the evaluation performance in a format that can be easily pasted."""

    delimiter_per_target: ClassVar[dict[PasteTarget, str]] = {
        "paper": " & ",
        "short": " & ",
        "full": " & ",
        "excel": ",",
    }
    success_prefix: ClassVar[str] = "success/"
    levels: ClassVar[list[Level]] = ["L1", "L2", "L3", "L4"]
    textual_override: ClassVar[str] = "{---}"
    level_format_prefix: ClassVar[str] = r"\bfseries "

    def __init__(
        self,
        *,
        target: PasteTarget,
        force_textual: bool = False,
        strict_time_limit: bool = False,
        compute_delta: bool = False,
        print_full: bool = True,
    ) -> None:
        self.target = target
        self.delimiter = self.delimiter_per_target[target]
        self.force_textual = force_textual
        self.strict_time_limit = strict_time_limit
        self.compute_delta = compute_delta
        self.print_full = print_full

    def __call__(self, run_ids: list[str]) -> None:
        """Print the performance in a format that can be easily pasted."""
        to_print = []
        for run_id in run_ids:
            try:
                full = self.process_run(run_id)
            except pl.ColumnNotFoundError:
                logger.error(f"Run {run_id} does not have the required columns.")
                continue
            to_print.extend(full)

        if self.target == "full" and self.print_full:
            for line in to_print:
                console.print(line)

    def process_run(self, run_id: str) -> list[str]:
        """Print the performance for a single run."""
        for_printing = []

        run = get_run(run_id)
        is_textual = (
            "textual" in run.name.lower()
            or "ObjText".lower() in run.name.lower()
            or self.force_textual
        )
        console.print("Is Textual:", is_textual)

        for_printing.append(
            r"\multicolumn{18}{@{}l}{\textit{" + get_latex_model_name(run) + r"}} \\"
        )

        performances = self.get_evaluation_performance(run)

        console.print("Compute Delta:", self.compute_delta)
        if self.compute_delta:
            performances = self.compute_delta_performance(run, performances)

        if self.target in {"paper", "short"}:
            self.print_for_paper(performances, is_textual=is_textual)
        if self.target == "excel":
            self.print_for_excel(performances, is_textual=is_textual)
        if self.target == "full":
            full = self.print_for_paper_draft(performances, is_textual=is_textual)
            for_printing.extend(full)

        return for_printing

    def get_evaluation_performance(self, run: Run) -> dict[Level, dict[int, Decimal]]:
        """Get the evaluation performance from WandB."""
        console.print("Is Strict Time Limit:", self.strict_time_limit)
        if self.strict_time_limit:
            return self.get_strict_time_limit_evaluation_performance(run)
        return self.get_default_evaluation_performance(run)

    def get_default_evaluation_performance(self, run: Run) -> dict[Level, dict[int, Decimal]]:
        """Get the evaluation performance from WandB."""
        performance_per_level = {}
        # Get all the success metrics per partition
        for level in self.levels:
            task_success = {
                int(key[-2:]): Decimal(success * 100).quantize(Decimal("1.0"))
                for key, success in sorted(run.summary.items())
                if key.startswith(f"{self.success_prefix}{level}")
            }
            performance_per_level[level] = task_success

        return performance_per_level

    def get_strict_time_limit_evaluation_performance(
        self, run: Run
    ) -> dict[Level, dict[int, Decimal]]:
        """Get the strict evaluation performance from WandB."""
        episodes_path = download_episodes_from_run(run)
        episodes_df = process_episodes_table_from_wandb(episodes_path)
        per_task_performance = compute_per_task_performance(episodes_df)
        performances = per_task_performance.select(
            pl.col("partition").cast(str).add("L").str.reverse().alias("partition"),
            pl.col("task"),
            pl.col("percentage_successful_strict").mul(100).round(1).alias("success"),
        ).to_dicts()

        performance_per_level = {}
        for success in performances:
            if success["partition"] not in performance_per_level:
                performance_per_level[success["partition"]] = {}

            performance_per_level[success["partition"]][success["task"]] = Decimal(
                success["success"]
            )

        return performance_per_level

    def compute_delta_performance(
        self, run: Run, evaluation_performance: dict[Level, dict[int, Decimal]]
    ) -> dict[Level, dict[int, Decimal]]:
        """Compute the delta performance."""
        baseline_run_id = get_baseline_run_id(run)
        baseline_run = wandb.Api().run(f"pyop/cogelot-evaluation/{baseline_run_id}")
        baseline_performance = self.get_evaluation_performance(baseline_run)
        delta_performance = {}
        for level in self.levels:
            delta_performance[level] = {
                task: performance - baseline_performance[level][task]
                for task, performance in evaluation_performance[level].items()
            }
        return delta_performance

    def print_for_paper(
        self, performances: dict[Level, dict[int, Decimal]], *, is_textual: bool
    ) -> None:
        """Print the performance in a format that can be easily pasted in LaTeX."""
        averages = []
        for task_success in performances.values():
            if is_textual:
                task_success = {  # noqa: PLW2901
                    task_num: task_value
                    if Task(task_num - 1) not in TextualDescriptionTransform.tasks_to_avoid
                    else self.textual_override
                    for task_num, task_value in task_success.items()
                }
            average = statistics.mean(
                task_value
                for task_value in task_success.values()
                if isinstance(task_value, Decimal)
            )
            averages.append(average.quantize(Decimal("1.0")))

        # console.print(" & ".join(self.levels))
        console.print(" & ".join(map(str, averages)), r"\\")

    def print_for_paper_draft(
        self, performances: dict[Level, dict[int, Decimal]], *, is_textual: bool
    ) -> list[str]:
        """Print the performance in a format that can be easily pasted in LaTeX."""
        printing_lines = []

        for level, task_success in performances.items():
            task_success = {  # noqa: PLW2901
                task.value + 1: task_success.get(task.value + 1, self.textual_override)
                for task in Task
            }
            if is_textual:
                task_success = {  # noqa: PLW2901
                    task_num: task_value
                    if Task(task_num - 1) not in TextualDescriptionTransform.tasks_to_avoid
                    else self.textual_override
                    for task_num, task_value in task_success.items()
                }

            average = statistics.mean(
                task_value
                for task_value in task_success.values()
                if isinstance(task_value, Decimal)
            )
            print_line = self.delimiter.join(map(str, task_success.values()))
            print_line += self.delimiter + str(average.quantize(Decimal("1.0")))
            print_line = f"{self.level_format_prefix}{level} & {print_line} " + r"\\"
            printing_lines.append(print_line)

            if not self.print_full:
                console.print(print_line)

        return printing_lines

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


def print_performance(
    run_ids: list[str],
    *,
    target: Annotated[str, typer.Option()],
    force_textual: Annotated[bool, typer.Option()] = False,
    strict_time_limit: Annotated[bool, typer.Option()] = False,
    compute_delta: Annotated[bool, typer.Option()] = False,
) -> None:
    """Get the evaluation performance from WandB for easy pasting."""
    assert target in EvaluationPerformancePrinter.delimiter_per_target
    printer = EvaluationPerformancePrinter(
        target=target,
        force_textual=force_textual,
        strict_time_limit=strict_time_limit,
        compute_delta=compute_delta,
    )
    printer(run_ids)


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
