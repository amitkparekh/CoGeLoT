from pathlib import Path
from typing import cast

import dill as pickle
from loguru import logger
from torchdata.datapipes.iter import FileLister, IterableWrapper, IterDataPipe, Multiplexer

from cogelot.structures.model import PreprocessedInstance
from cogelot.structures.vima import Task


def load_preprocessed_instances(
    preprocessed_data_dir: Path,
) -> IterDataPipe[PreprocessedInstance]:
    """Load preprocessed data from the disk."""
    datapipe = (
        FileLister(root=str(preprocessed_data_dir))
        .open_files("rb")
        .map(lambda x: x[1])
        .map(pickle.load)
    )
    return cast(IterDataPipe[PreprocessedInstance], datapipe)


def batch_datapipe(
    preprocessed_datapipe: IterDataPipe[PreprocessedInstance], batch_size: int
) -> IterDataPipe[list[PreprocessedInstance]]:
    """Batch the preprocessed data."""
    return (
        preprocessed_datapipe.shuffle()
        .sharding_filter()
        .batch(batch_size=batch_size, drop_last=True)
    )


def get_all_tasks_in_dataset(
    preprocessed_datapipe: IterDataPipe[PreprocessedInstance],
) -> set[Task]:
    """Get all the tasks in the dataset."""
    return set(preprocessed_datapipe.map(lambda instance: instance.task))


def split_instances_per_task(
    preprocessed_datapipe: IterDataPipe[PreprocessedInstance],
) -> list[IterDataPipe[PreprocessedInstance]]:
    """Split the datapipe into one for each type of task."""
    all_tasks_in_datapipe = get_all_tasks_in_dataset(preprocessed_datapipe)

    # Create an id for each task
    id_per_task: dict[Task, int] = {
        task: index for index, task in enumerate(all_tasks_in_datapipe)
    }

    # Split the datapipe into one datapipe per task
    instances_per_task = preprocessed_datapipe.demux(
        num_instances=len(all_tasks_in_datapipe),
        classifier_fn=lambda instance: id_per_task[instance.task],
    )

    return instances_per_task


def create_validation_split(
    preprocessed_datapipe: IterDataPipe[PreprocessedInstance],
    num_val_instances: int,
    seed: int = 0,
) -> tuple[IterDataPipe[PreprocessedInstance], IterDataPipe[PreprocessedInstance]]:
    """Split the preprocessed data into a validation set and a training set."""
    num_tasks_in_dataset = len(get_all_tasks_in_dataset(preprocessed_datapipe))
    num_val_instances_per_task = num_val_instances // num_tasks_in_dataset

    # If less than 1 validation instance desired per task, return the original datapipe and an
    # empty validation datapipe
    if num_val_instances_per_task < 1 or num_val_instances == 0:
        logger.warning(
            "Less than 1 validation instance per task, returning empty validation datapipe."
        )
        return preprocessed_datapipe, cast(IterDataPipe[PreprocessedInstance], IterableWrapper([]))

    # Create a list of training and validation datapipes
    training_instance_datapipes: list[IterDataPipe[PreprocessedInstance]] = []
    validation_instance_datapipes: list[IterDataPipe[PreprocessedInstance]] = []

    # Split the datapipe into one datapipe per task, and create the split for each task
    for task_instances_datapipe in split_instances_per_task(preprocessed_datapipe):
        total_num_instances = len(list(task_instances_datapipe))

        training, validation = task_instances_datapipe.random_split(
            total_length=total_num_instances,
            weights={
                "train": total_num_instances - num_val_instances_per_task,
                "valid": num_val_instances_per_task,
            },
            seed=seed,
        )

        training_instance_datapipes.append(training)
        validation_instance_datapipes.append(validation)

    # Merge the list of datapipe back into one datapipe for each split
    train_instances = Multiplexer(*training_instance_datapipes)
    validation_instances = Multiplexer(*validation_instance_datapipes)
    train_instances = cast(IterDataPipe[PreprocessedInstance], train_instances)
    validation_instances = cast(IterDataPipe[PreprocessedInstance], validation_instances)

    return train_instances, validation_instances
