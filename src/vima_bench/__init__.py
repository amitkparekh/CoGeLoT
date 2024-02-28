from .make import make
from .tasks import ALL_PARTITIONS, ALL_TASKS

__all__ = ["make", "ALL_TASKS", "ALL_PARTITIONS"]

ALL_TASKS = list(ALL_TASKS.keys())
