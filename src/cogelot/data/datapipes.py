from pathlib import Path
from typing import NamedTuple, cast

import dill as pickle
from torchdata.datapipes.iter import FileLister, IterableWrapper, IterDataPipe

from cogelot.data.normalize import (
    create_vima_instance_from_instance_dir,
    get_all_instance_directories,
)
from cogelot.structures.model import PreprocessedInstance
from cogelot.structures.vima import VIMAInstance


def normalize_raw_data(raw_data_root: Path) -> IterDataPipe[VIMAInstance]:
    """Create a datapipe to normalize all the raw data."""
    normalize_raw_datapipe = (
        IterableWrapper(list(get_all_instance_directories(raw_data_root)))
        .sharding_filter()
        .map(create_vima_instance_from_instance_dir)
    )

    return cast(IterDataPipe[VIMAInstance], normalize_raw_datapipe)


def cache_normalized_data(
    normalized_data_pipe: IterDataPipe[VIMAInstance], normalized_data_root: Path
) -> IterDataPipe[Path]:
    """Create a datapipe to cache the normalized data to disk."""
    return (
        normalized_data_pipe.map(
            lambda instance: (normalized_data_root.joinpath(instance.file_name), instance.json())
        )
        .save_to_disk(mode="w")
        .map(Path)
    )


def load_cached_normalized_data(normalized_data_dir: Path) -> IterDataPipe[VIMAInstance]:
    """Load normalized data from the disk."""
    datapipe = (
        FileLister(root=str(normalized_data_dir))
        .sharding_filter()
        .map(lambda path: VIMAInstance.parse_file(path))
    )
    return cast(IterDataPipe[VIMAInstance], datapipe)


def cache_preprocessed_data(
    preprocessed_datapipe: IterDataPipe[PreprocessedInstance], preprocessed_data_dir: Path
) -> IterDataPipe[Path]:
    """Create a datapipe to cache the preprocessed data to disk."""
    return (
        preprocessed_datapipe.enumerate()
        .map(
            lambda enumerated_tuple: (
                (
                    preprocessed_data_dir.joinpath(f"{enumerated_tuple[0]}.pt"),
                    pickle.dumps(enumerated_tuple[1]),
                )
            )
        )
        .save_to_disk(mode="wb")
    )


def load_cached_preprocessed_data(
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


class DatasetSplit(NamedTuple):
    """A named tuple to hold the train and validation splits."""

    train: IterDataPipe[PreprocessedInstance]
    valid: IterDataPipe[PreprocessedInstance]


def create_validation_split(
    preprocessed_datapipe: IterDataPipe[PreprocessedInstance],
    num_val_instances: int,
    seed: int = 0,
) -> tuple[IterDataPipe[PreprocessedInstance], IterDataPipe[PreprocessedInstance]]:
    """Split the preprocessed data into a validation set and a training set."""
    total_num_instances = len(list(preprocessed_datapipe))

    train, valid = preprocessed_datapipe.random_split(
        total_length=total_num_instances,
        weights={"train": total_num_instances - num_val_instances, "valid": num_val_instances},
        seed=seed,
    )

    return train, valid


def batch_datapipe(
    preprocessed_datapipe: IterDataPipe[PreprocessedInstance], batch_size: int
) -> IterDataPipe[list[PreprocessedInstance]]:
    """Batch the preprocessed data."""
    return (
        preprocessed_datapipe.shuffle()
        .sharding_filter()
        .batch(batch_size=batch_size, drop_last=True)
    )
