import itertools
from typing import Any, Self

from torch.utils.data import Dataset

from cogelot.structures.model import EvaluationEpisode
from cogelot.structures.vima import Partition, Task
from vima_bench.tasks import get_partition_to_specs


def get_every_partition_task_combination(
    partition_to_specs: dict[str, dict[str, Any]] | None = None,
) -> list[EvaluationEpisode]:
    """Get every partition-task combo from VIMA's `PARTITION_TO_SPECS`."""
    partition_to_specs = partition_to_specs or get_partition_to_specs()
    raw_instances_per_task_per_partition = partition_to_specs["test"]
    episodes = [
        EvaluationEpisode(partition=Partition[partition], task=Task[task])
        for partition, tasks in raw_instances_per_task_per_partition.items()
        for task in tasks
    ]
    return episodes


class VIMAEvaluationDataset(Dataset[EvaluationEpisode]):
    """Create the evaluation dataset."""

    def __init__(self, instances: list[EvaluationEpisode]) -> None:
        super().__init__()
        self._instances = instances

    def __len__(self) -> int:
        """Return the number of instances in the dataset."""
        return len(self._instances)

    def __getitem__(self, index: int) -> EvaluationEpisode:
        """Return the instance at the given index."""
        return self._instances[index]

    def filter_for_tasks(self, *tasks: Task) -> None:
        """Filter all the instances, keeping only the given task."""
        self._instances = [instance for instance in self._instances if instance.task in tasks]

    def filter_for_partitions(self, *partitions: Partition) -> None:
        """Filter all the instances, keeping the desired partition."""
        self._instances = [
            instance for instance in self._instances if instance.partition in partitions
        ]

    @classmethod
    def from_partition_to_specs(
        cls,
        partition_to_specs: dict[str, dict[str, Any]] | None = None,
        num_repeats_per_episode: int = 100,
    ) -> Self:
        """Instantiate from VIMA's `PARTITION_TO_SPECS`."""
        episodes = get_every_partition_task_combination(partition_to_specs)

        instances = itertools.chain.from_iterable(
            itertools.repeat(episode, num_repeats_per_episode) for episode in episodes
        )

        return cls(list(instances))
