from functools import partial
from typing import Annotated, Optional

import typer
from loguru import logger

from cogelot.common.settings import Settings
from cogelot.data.transforms import RewordPromptTransform
from cogelot.entrypoints.preprocess_instances import load_parsed_datasets_for_each_task
from cogelot.structures.vima import Task, VIMAInstance


def create_reworded_dataset_per_task(
    old_dataset_variant: Annotated[str, typer.Argument(help="Old dataset variant.")],
    new_dataset_variant: Annotated[str, typer.Argument(help="New dataset variant.")] = "reworded",
    num_workers: Annotated[int, typer.Option(help="Number of workers.")] = 1,
    writer_batch_size: Annotated[
        int,
        typer.Option(help="Batch size when creating the dataset for each task."),
    ] = 500,
    task_index_filter: Annotated[
        Optional[int], typer.Option(min=Task.minimum(), max=Task.maximum())  # noqa: UP007
    ] = None,
) -> None:
    """Augment an existing dataset variant into the reworded one.

    This seems to just be a slow thing to run, and something that is very RAM hungry. I'm not sure
    on how to optimise this but I'm sure that going to and from a strictly-validated Pydantic model
    probably doesn't help. That said, I'm not sure how to get around that, so we just wait.

    If you're able to break apart the tasks into multiple jobs, then I'd recommend a RAM size of
    200G for 20 workers. That's what I did. Each task was processed in about 2-3 hours.
    """
    # Make sure the dataset variants are valid
    assert old_dataset_variant in {"original", "keep_null_action"}
    assert new_dataset_variant in {"reworded", "reworded_keep_null_action"}
    assert (old_dataset_variant == "original" and new_dataset_variant == "reworded") or (
        old_dataset_variant == "keep_null_action"
        and new_dataset_variant == "reworded_keep_null_action"
    )

    old_parsed_hf_dataset_dir = Settings(dataset_variant=old_dataset_variant).parsed_hf_dataset_dir
    new_parsed_hf_dataset_dir = Settings(dataset_variant="reworded").parsed_hf_dataset_dir
    new_parsed_hf_dataset_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading parsed dataset for each task...")
    dataset_per_task_iterator = load_parsed_datasets_for_each_task(
        old_parsed_hf_dataset_dir, task_index_filter=task_index_filter
    )

    instance_transformer = RewordPromptTransform()
    for task, dataset in dataset_per_task_iterator:  # noqa: WPS426
        if task_index_filter is not None and task_index_filter != task.value:
            continue

        logger.info(f"Augmenting data for {task}")
        augmented_dataset = dataset.map(
            partial(
                lambda example, instance_transformer: instance_transformer(
                    VIMAInstance.model_validate(example)
                ).model_dump(),
                instance_transformer=instance_transformer,
            ),
            num_proc=num_workers,
            writer_batch_size=writer_batch_size,
        )
        logger.info(f"Saving dataset for {task}...")
        augmented_dataset.save_to_disk(
            new_parsed_hf_dataset_dir.joinpath(task.name),
            num_shards=Settings().num_shards,
            num_proc=num_workers,
        )


if __name__ == "__main__":
    create_reworded_dataset_per_task(
        old_dataset_variant="original", task_index_filter=5, num_workers=20
    )
